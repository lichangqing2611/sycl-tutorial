#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>


const int dim2 = 1;
const int dim1 = 1;
const int dim0 = 1024 * 1024 * 2;
const int wg = 512;

/******************************************** addition1 **************************************************/
inline void addition1_kernel(const sycl::nd_item<3> &item, const float *g_idata, float *g_odata) {
    uint64_t tid = item.get_local_id(2);
    uint64_t gid = item.get_group(2) * item.get_local_range(2) + tid;
    g_odata[gid] = g_idata[gid] + 0.0001f;
}

sycl::event addition1(sycl::queue &q, const float *input_gpu, float *output_gpu) {
    auto ev_exec = q.submit([&](sycl::handler &cgh) {
        sycl::range<3> grid_size(dim2, dim1, dim0 / wg);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_addition1>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                addition1_kernel(item, input_gpu, output_gpu);
        });
    });

    q.wait();

    return ev_exec;
}

/******************************************** addition2 **************************************************/
inline void addition2_kernel(const sycl::nd_item<3> &item, const float *g_idata, const int n, float *g_odata) {
    uint64_t tid = item.get_local_id(2);
    uint64_t block_size = item.get_local_range(2);
    uint64_t gid = item.get_group(2) * item.get_local_range(2) + tid;
    while (gid < n) {
        g_odata[gid] = g_idata[gid] + 0.0001f;
        gid += block_size;
    }
}

sycl::event addition2(sycl::queue &q, const float *input_gpu, float *output_gpu) {
    auto ev_exec = q.submit([&](sycl::handler &cgh) {
        sycl::range<3> grid_size(dim2, dim1, 1);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_addition2>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                addition2_kernel(item, input_gpu, dim0, output_gpu);
        });
    });

    q.wait();

    return ev_exec;
}

/******************************************** addition3 **************************************************/
inline void addition3_kernel(const sycl::nd_item<3> &item, const float *g_idata, const int n, float *g_odata) {
    uint64_t tid = item.get_local_id(2);
    uint64_t block_size = item.get_local_range(2);
    uint64_t gid = item.get_group(2) * item.get_local_range(2) + tid;
    uint64_t start_per_thread = gid * n / block_size;
    uint64_t end_per_thread = gid * n / block_size + n / block_size;
    while (start_per_thread < end_per_thread) {
        g_odata[start_per_thread] = g_idata[start_per_thread] + 0.0001f;
        start_per_thread++;
    }
}

sycl::event addition3(sycl::queue &q, const float *input_gpu, float *output_gpu) {
    auto ev_exec = q.submit([&](sycl::handler &cgh) {
        sycl::range<3> grid_size(dim2, dim1, 1);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_addition3>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                addition3_kernel(item, input_gpu, dim0, output_gpu);
        });
    });

    q.wait();

    return ev_exec;
}

/******************************************** addition4 **************************************************/
inline void addition4_kernel(const sycl::nd_item<1> &item, const float *g_idata, float *g_odata) {
    uint64_t tid = item.get_local_id(0);
    uint64_t gid = item.get_group(0) * item.get_local_range(0) + tid;
    g_odata[gid] = g_idata[gid] + 0.0001f;
}

sycl::event addition4(sycl::queue &q, const float *input_gpu, float *output_gpu) {
    auto ev_exec = q.submit([&](sycl::handler &cgh) {
        sycl::range<1> grid_size(dim0 / wg);
        sycl::range<1> work_group_size(wg);

        cgh.parallel_for<class kernel_addition4>(
            sycl::nd_range<1>(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                addition4_kernel(item, input_gpu, output_gpu);
        });
    });

    q.wait();

    return ev_exec;
}

/******************************************** addition5 **************************************************/
sycl::event addition5(sycl::queue &q, const float *input_gpu, float *output_gpu) {
    auto ev_exec = q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class kernel_addition5>(sycl::range<1>(dim0), [=](sycl::id<1> idx) {
            output_gpu[idx] = input_gpu[idx] + 0.0001f;
        });
    });

    q.wait();

    return ev_exec;
}

int main() {
    sycl::device dev = sycl::device::get_devices().at(3);
    const sycl::property_list queue_prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
    sycl::queue q{dev, queue_prop_list};

    std::vector<float> input(dim2 * dim1 * dim0, 0.0f);
    std::vector<float> output(dim2 * dim1 * dim0, 0.0f);

    for (int i = 0; i < input.size(); ++i) {
        input[i] = 0.0001f * (i % 117);
    }

    float *input_gpu = sycl::malloc_device<float>(input.size(), q);
    float *output_gpu = sycl::malloc_device<float>(output.size(), q);
    q.memcpy(input_gpu, input.data(), input.size() * sizeof(float)).wait();
    q.memset(output_gpu, 0, input.size() * sizeof(float)).wait();

    {
        addition1(q, input_gpu, output_gpu);
        addition1(q, input_gpu, output_gpu);
        auto start_time = std::chrono::high_resolution_clock::now();
        sycl::event ev = addition1(q, input_gpu, output_gpu);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("addition1 latency: %.6f ms, bandwidth: %.6f GB/sec \n", during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (during_ms / 1000));

        float ev_during_ms = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                              ev.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
        printf("addition1 latency: %.6f ms, bandwidth: %.6f GB/sec \n", ev_during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (ev_during_ms / 1000));

        q.memcpy(output.data(), output_gpu, output.size() * sizeof(float)).wait();
        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        addition2(q, input_gpu, output_gpu);
        addition2(q, input_gpu, output_gpu);
        auto start_time = std::chrono::high_resolution_clock::now();
        sycl::event ev = addition2(q, input_gpu, output_gpu);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("addition2 latency: %.6f ms, bandwidth: %.6f GB/sec \n", during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (during_ms / 1000));

        float ev_during_ms = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                              ev.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
        printf("addition2 latency: %.6f ms, bandwidth: %.6f GB/sec \n", ev_during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (ev_during_ms / 1000));

        q.memcpy(output.data(), output_gpu, output.size() * sizeof(float)).wait();
        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        addition3(q, input_gpu, output_gpu);
        addition3(q, input_gpu, output_gpu);
        auto start_time = std::chrono::high_resolution_clock::now();
        sycl::event ev = addition3(q, input_gpu, output_gpu);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("addition3 latency: %.6f ms, bandwidth: %.6f GB/sec \n", during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (during_ms / 1000));

        float ev_during_ms = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                              ev.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
        printf("addition3 latency: %.6f ms, bandwidth: %.6f GB/sec \n", ev_during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (ev_during_ms / 1000));

        q.memcpy(output.data(), output_gpu, output.size() * sizeof(float)).wait();
        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        addition4(q, input_gpu, output_gpu);
        addition4(q, input_gpu, output_gpu);
        auto start_time = std::chrono::high_resolution_clock::now();
        sycl::event ev = addition4(q, input_gpu, output_gpu);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("addition4 latency: %.6f ms, bandwidth: %.6f GB/sec \n", during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (during_ms / 1000));

        float ev_during_ms = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                              ev.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
        printf("addition4 latency: %.6f ms, bandwidth: %.6f GB/sec \n", ev_during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (ev_during_ms / 1000));

        q.memcpy(output.data(), output_gpu, output.size() * sizeof(float)).wait();
        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        addition5(q, input_gpu, output_gpu);
        addition5(q, input_gpu, output_gpu);
        auto start_time = std::chrono::high_resolution_clock::now();
        sycl::event ev = addition5(q, input_gpu, output_gpu);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("addition5 latency: %.6f ms, bandwidth: %.6f GB/sec \n", during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (during_ms / 1000));

        float ev_during_ms = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                              ev.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
        printf("addition5 latency: %.6f ms, bandwidth: %.6f GB/sec \n", ev_during_ms, ((float)input.size() * sizeof(float) * 2 / 1024 / 1024 / 1024) / (ev_during_ms / 1000));

        q.memcpy(output.data(), output_gpu, output.size() * sizeof(float)).wait();
        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    float sum_cpu = 0.0f;
    for (int i = 0; i < input.size(); ++i) {
        sum_cpu += input[i] + 0.0001;
    }
    std::cout << "cpu sum: " << sum_cpu << std::endl;

    return 0;
}
