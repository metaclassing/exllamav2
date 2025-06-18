#include "ext_stloader.h"
#include "cpp/util.h"

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>

#define STLOADER_COPY_THREADS 4

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
    }

    // Synchronization

    Py_BEGIN_ALLOW_THREADS

    volatile bool load_failed = false;
    bool done_loading = false;
    std::mutex mtx;
    std::deque<std::pair<size_t, size_t>> dq;
    std::condition_variable cv;

    // Load chunks

    auto load_worker = [&] (size_t pos_a)
    {
        size_t blocks_processed = 0;
        FILE* file = fopen(filename, "rb");
        if (!file) goto error;

        while (pos_a < size && !load_failed)
        {
            size_t pos_b = pos_a + STLOADER_BLOCK_SIZE;
            if (pos_b > size) pos_b = size;

            #ifdef __linux__
                ssize_t br = pread(fileno(file), load_buffer + pos_a, pos_b - pos_a, offset + pos_a);
                if (br != pos_b - pos_a) goto error;
                int sr = fseek(file, offset + pos_a, SEEK_SET);
            #else
                int sr = _fseeki64(file, static_cast<__int64>(offset + pos_a), SEEK_SET);
                if (sr) goto error;
                size_t br = fread(load_buffer + pos_a, 1, pos_b - pos_a, file);
                if (br != pos_b - pos_a) goto error;
            #endif

            {
                std::lock_guard<std::mutex> lock(mtx);
                dq.push_back(std::pair<size_t, size_t>(pos_a, pos_b));
                cv.notify_one();
            }

            pos_a += STLOADER_THREADS * STLOADER_BLOCK_SIZE;
            blocks_processed++;
        }

        printf("[stloader] load_worker thread %zu processed %zu blocks\n", pos_a / (STLOADER_THREADS * STLOADER_BLOCK_SIZE), blocks_processed);

        fclose(file);
        return;

        error:
        if (file && ferror(file))
            printf("Error reading file: %s (errno: %d)\n", strerror(errno), errno);
        load_failed = true;
    };

    // Copy chunks to device (parallelized)

    auto copy_worker = [&] (int stream_idx)
    {
        size_t blocks_copied = 0;
        cudaSetDevice(device.value().index());
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        while (true)
        {
            size_t pos_a = 0, pos_b = 0;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&dq, &load_failed, &done_loading] { return !dq.empty() || load_failed || done_loading; });
                if (load_failed) break;
                if (dq.empty()) {
                    if (done_loading) break;
                    continue;
                }

                auto pop = dq.front();
                dq.pop_front();
                pos_a = std::get<0>(pop);
                pos_b = std::get<1>(pop);

                // Optionally, coalesce adjacent blocks for efficiency (as before)
                while (!dq.empty() && std::get<0>(dq.front()) == pos_b)
                {
                    pop = dq.front();
                    dq.pop_front();
                    pos_b = std::get<1>(pop);
                }
            }

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
                fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(cr));
                load_failed = true;
                break;
            }
            blocks_copied++;
        }
        printf("[stloader] copy_worker thread %d copied %zu blocks\n", stream_idx, blocks_copied);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    };

    std::vector<std::thread> threads;
    size_t num_load_workers = 0;
    for (size_t i = 0; i < STLOADER_THREADS && i * STLOADER_BLOCK_SIZE < size; ++i) {
        threads.emplace_back(load_worker, i * STLOADER_BLOCK_SIZE);
        num_load_workers++;
    }

    std::vector<std::thread> copy_threads;
    if (cuda_buffer)
    {
        for (int i = 0; i < STLOADER_COPY_THREADS; ++i)
            copy_threads.emplace_back(copy_worker, i);
    }

    // Wait for all load workers to finish
    for (size_t i = 0; i < num_load_workers; ++i)
        threads[i].join();

    // Signal copy workers that loading is done
    {
        std::lock_guard<std::mutex> lock(mtx);
        done_loading = true;
        cv.notify_all();
    }

    // Wait for all copy workers to finish
    for (auto& thread : copy_threads)
        thread.join();

    TORCH_CHECK(!load_failed, "I/O error reading tensor");

    if (!target_cpu)
    {
        free(load_buffer);
        cudaDeviceSynchronize();
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
