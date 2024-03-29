#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>


const int dim2 = 1;
const int dim1 = 1;
const int dim0 = 1024 * 4;
const int wg = 1024;

inline void reduction1_kernel(const sycl::nd_item<3> &item,
                        const sycl::accessor<float, 1, sycl::access::mode::read> &g_idata,
                        const sycl::accessor<float, 1, sycl::access::mode::write> &g_odata,
                        float *sdata,
                        const sycl::stream &out) {
    unsigned int tid = item.get_local_id(2);
    unsigned int gid = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    sdata[tid] = g_idata[gid];
    item.barrier(sycl::access::fence_space::local_space);

    // do reduction in shared mem
    for(unsigned int s = 1; s < item.get_local_range(2); s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[item.get_group(2)] = sdata[0];
}

void reduction1(sycl::queue &q, const std::vector<float> &input, std::vector<float> &output) {
    sycl::buffer<float, 1> bufferI(input.data(), sycl::range<1>(input.size()));
    sycl::buffer<float, 1> bufferO(output.data(), sycl::range<1>(output.size()));

    q.submit([&](sycl::handler &cgh) {
        auto out = sycl::stream(10240, 7680, cgh);

        auto accessorI = bufferI.get_access<sycl::access::mode::read>(cgh);
        auto accessorO = bufferO.get_access<sycl::access::mode::write>(cgh);
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(wg), cgh);

        sycl::range<3> grid_size(dim2, dim1, dim0 / wg);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_reduction1>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                reduction1_kernel(item, accessorI, accessorO, sdata.get_pointer(), out);
        });
    }).wait();
}

int main() {
    sycl::queue q;
    std::vector<float> input(dim2 * dim1 * dim0);
    std::vector<float> output(dim0 / wg);

    for (int i = 0; i < input.size(); ++i) {
        input[i] = 0.0001f * i;
    }

    for (int i = 0; i < output.size(); ++i) {
        output[i] = 0.0f;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    reduction1(q, input, output);
    auto end_time = std::chrono::high_resolution_clock::now();
    float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    printf("reduction1 latency: %.6f ms\n", during_ms);

    std::cout << "gpu sum: ";
    float sum_gpu = 0.0f;
    for (int i = 0; i < output.size(); ++i) {
        std::cout << output[i] << ", ";
        sum_gpu += output[i];
    }
    std::cout << "+= " << sum_gpu << std::endl;

    float sum_cpu = 0.0f;
    for (int i = 0; i < input.size(); ++i) {
        sum_cpu += input[i];
    }
    std::cout << "cpu sum: " << sum_cpu << std::endl;

    return 0;
}
