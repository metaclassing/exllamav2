#include "ext_stloader.h"
#include "cpp/util.h"

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <atomic>

void stloader_read
(
    const char* filename,
    size_t offset,
    size_t size,
    torch::Tensor target
)
{
    c10::optional<torch::Device> device = torch::device_of(target);
    bool target_cpu = (device.has_value() && device->type() == torch::kCPU);
    cudaStream_t stream;

    // Buffers
    uint8_t* load_buffer;
    uint8_t* cuda_buffer;
    if (target_cpu)
    {
        load_buffer = (uint8_t*) target.data_ptr();
        cuda_buffer = nullptr;
    }
    else
    {
        load_buffer = (uint8_t*) malloc(size);
        TORCH_CHECK(load_buffer, "Can't allocate buffer for tensor");
        cuda_buffer = (uint8_t*) target.data_ptr();
        cudaSetDevice(device.value().index());
        stream = at::cuda::getCurrentCUDAStream(device.value().index()).stream();
    }

    // Synchronization
    Py_BEGIN_ALLOW_THREADS

    std::atomic<bool> load_failed{false};
    std::atomic<bool> load_complete{false};
    std::mutex mtx;
    std::deque<std::pair<size_t, size_t>> dq;
    std::condition_variable cv;
    
    // Track total blocks for copy worker termination
    size_t total_blocks = DIVIDE(size, STLOADER_BLOCK_SIZE);
    std::atomic<size_t> blocks_produced{0};

    // Load chunks - each worker gets its own file handle
    auto load_worker = [&] (size_t thread_id)
    {
        FILE* file = fopen(filename, "rb");
        if (!file) {
            load_failed = true;
            cv.notify_all();
            return;
        }

        size_t pos_a = thread_id * STLOADER_BLOCK_SIZE;
        
        while (pos_a < size && !load_failed)
        {
            size_t pos_b = pos_a + STLOADER_BLOCK_SIZE;
            if (pos_b > size) pos_b = size;

            #ifdef __linux__
                ssize_t br = pread(fileno(file), load_buffer + pos_a, pos_b - pos_a, offset + pos_a);
                if (br != (ssize_t)(pos_b - pos_a)) {
                    if (!load_failed) {
                        printf("Error reading file: %s (errno: %d)\n", strerror(errno), errno);
                        load_failed = true;
                    }
                    break;
                }
            #else
                int sr = _fseeki64(file, static_cast<__int64>(offset + pos_a), SEEK_SET);
                if (sr) {
                    if (!load_failed) {
                        printf("Error seeking file: %s (errno: %d)\n", strerror(errno), errno);
                        load_failed = true;
                    }
                    break;
                }
                size_t br = fread(load_buffer + pos_a, 1, pos_b - pos_a, file);
                if (br != pos_b - pos_a) {
                    if (!load_failed) {
                        printf("Error reading file: %s (errno: %d)\n", strerror(errno), errno);
                        load_failed = true;
                    }
                    break;
                }
            #endif

            // Add to queue atomically
            {
                std::lock_guard<std::mutex> lock(mtx);
                dq.push_back(std::pair<size_t, size_t>(pos_a, pos_b));
                blocks_produced++;
                cv.notify_one();
            }

            pos_a += STLOADER_THREADS * STLOADER_BLOCK_SIZE;
        }

        fclose(file);
        
        // Notify that this worker is done
        cv.notify_all();
    };

    // Copy chunks to device
    auto copy_worker = [&] ()
    {
        if (!cuda_buffer) return; // No copying needed for CPU tensors
        
        cudaSetDevice(device.value().index());
        size_t blocks_processed = 0;

        while (blocks_processed < total_blocks && !load_failed)
        {
            size_t pos_a, pos_b;
            
            // Wait for data or completion
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { 
                    return !dq.empty() || load_failed || 
                           (load_complete && blocks_produced == blocks_processed); 
                });

                if (dq.empty()) {
                    if (load_failed || (load_complete && blocks_produced == blocks_processed)) {
                        break;
                    }
                    continue;
                }

                auto pop = dq.front();
                dq.pop_front();
                pos_a = std::get<0>(pop);
                pos_b = std::get<1>(pop);

                // Coalesce contiguous blocks
                while (!dq.empty() && std::get<0>(dq.front()) == pos_b)
                {
                    pop = dq.front();
                    dq.pop_front();
                    pos_b = std::get<1>(pop);
                }
            }

            // Copy to GPU
            cudaError_t cr = cudaMemcpyAsync
            (
                cuda_buffer + pos_a,
                load_buffer + pos_a,
                pos_b - pos_a,
                cudaMemcpyHostToDevice,
                stream
            );
            
            if (cr != cudaSuccess)
            {
                fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(cr));
                load_failed = true;
                break;
            }
            
            blocks_processed++;
        }
    };

    // Start worker threads
    std::vector<std::thread> threads;
    
    // Start load workers
    for (size_t i = 0; i < STLOADER_THREADS && i * STLOADER_BLOCK_SIZE < size; ++i)
        threads.emplace_back(load_worker, i);
    
    // Start copy worker if needed
    if (cuda_buffer)
        threads.emplace_back(copy_worker);

    // Wait for all load workers to complete
    for (size_t i = 0; i < std::min(STLOADER_THREADS, size_t(threads.size())); ++i) {
        if (i < threads.size()) {
            threads[i].join();
        }
    }
    
    // Mark loading as complete
    load_complete = true;
    cv.notify_all();
    
    // Wait for copy worker if it exists
    if (cuda_buffer && threads.size() > STLOADER_THREADS) {
        threads.back().join();
    }

    TORCH_CHECK(!load_failed, "I/O error reading tensor");

    if (!target_cpu)
    {
        free(load_buffer);
        // Synchronize the specific stream instead of all streams
        cudaStreamSynchronize(stream);
    }

    Py_END_ALLOW_THREADS
}

void tensor_remap
(
    torch::Tensor tensor,
    torch::Tensor index
)
{
    TORCH_CHECK_SHAPES(tensor, 1, index, 0, 1);
    TORCH_CHECK_DTYPE(tensor, kInt);
    TORCH_CHECK_DTYPE(index, kInt);

    int rows = tensor.size(0);
    int cols = tensor.size(1);
    uint32_t* temp = (uint32_t*) calloc(cols, sizeof(int));
    uint32_t* a = (uint32_t*) tensor.data_ptr();
    uint32_t* idx = (uint32_t*) index.data_ptr();

    for (int r = 0; r < rows; ++r)
    {
        memcpy(temp, a, sizeof(uint32_t) * cols);
        for (int c = 0; c < cols; ++c)
        {
            *a++ = temp[idx[c]];
        }
    }
    free(temp);
}

void tensor_remap_4bit
(
    torch::Tensor tensor,
    torch::Tensor index
)
{
    TORCH_CHECK_SHAPES(index, 0, tensor, 1, 8);
    TORCH_CHECK_DTYPE(tensor, kInt);
    TORCH_CHECK_DTYPE(index, kInt);

    int rows = tensor.size(0);
    int cols = index.size(0);
    uint32_t* temp = (uint32_t*) calloc(cols / 8, sizeof(int));
    uint32_t* a = (uint32_t*) tensor.data_ptr();
    uint32_t* idx = (uint32_t*) index.data_ptr();

    for (int r = 0; r < rows; ++r)
    {
        memcpy(temp, a, sizeof(uint32_t) * cols / 8);
        for (int c = 0; c < cols;)
        {
            uint32_t rv = 0;
            for (int b = 0; b < 8; ++b, ++c)
            {
                uint32_t i = idx[c];
                uint32_t v = (temp[i / 8] >> ((i & 7) * 4) & 0x0f);
                rv |= v << (b * 4);
            }
            *a++ = rv;
        }
    }
    free(temp);
}
