#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>


const int dim2 = 1;
const int dim1 = 1;
const int dim0 = 1024 * 1024 * 2;
const int wg = 1024;

/******************************************** reduction1 **************************************************/
inline void reduction1_kernel(const sycl::nd_item<3> &item,
                        const sycl::accessor<float, 1, sycl::access::mode::read> &g_idata,
                        const sycl::accessor<float, 1, sycl::access::mode::write> &g_odata,
                        float *sdata,
                        const sycl::stream &out) {
    unsigned int tid = item.get_local_id(2);
    unsigned int gid = item.get_group(2) * item.get_local_range(2) + tid;
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

/******************************************** reduction2 **************************************************/
inline void reduction2_kernel(const sycl::nd_item<3> &item,
                        const sycl::accessor<float, 1, sycl::access::mode::read> &g_idata,
                        const sycl::accessor<float, 1, sycl::access::mode::write> &g_odata,
                        float *sdata,
                        const sycl::stream &out) {
    unsigned int tid = item.get_local_id(2);
    unsigned int gid = item.get_group(2) * item.get_local_range(2) + tid;
    sdata[tid] = g_idata[gid];
    item.barrier(sycl::access::fence_space::local_space);

    // do reduction in shared mem
    for(unsigned int s = 1; s < item.get_local_range(2); s *= 2) {
        int index = 2 * s * tid;
        if (index < item.get_local_range(2)) {
            sdata[index] += sdata[index + s];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[item.get_group(2)] = sdata[0];
}

void reduction2(sycl::queue &q, const std::vector<float> &input, std::vector<float> &output) {
    sycl::buffer<float, 1> bufferI(input.data(), sycl::range<1>(input.size()));
    sycl::buffer<float, 1> bufferO(output.data(), sycl::range<1>(output.size()));

    q.submit([&](sycl::handler &cgh) {
        auto out = sycl::stream(10240, 7680, cgh);

        auto accessorI = bufferI.get_access<sycl::access::mode::read>(cgh);
        auto accessorO = bufferO.get_access<sycl::access::mode::write>(cgh);
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(wg), cgh);

        sycl::range<3> grid_size(dim2, dim1, dim0 / wg);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_reduction2>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                reduction2_kernel(item, accessorI, accessorO, sdata.get_pointer(), out);
        });
    }).wait();
}

/******************************************** reduction3 **************************************************/
inline void reduction3_kernel(const sycl::nd_item<3> &item,
                        const sycl::accessor<float, 1, sycl::access::mode::read> &g_idata,
                        const sycl::accessor<float, 1, sycl::access::mode::write> &g_odata,
                        float *sdata,
                        const sycl::stream &out) {
    unsigned int tid = item.get_local_id(2);
    unsigned int gid = item.get_group(2) * item.get_local_range(2) + tid;
    sdata[tid] = g_idata[gid];
    item.barrier(sycl::access::fence_space::local_space);

    // do reduction in shared mem
    for (unsigned int s = item.get_local_range(2) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[item.get_group(2)] = sdata[0];
}

void reduction3(sycl::queue &q, const std::vector<float> &input, std::vector<float> &output) {
    sycl::buffer<float, 1> bufferI(input.data(), sycl::range<1>(input.size()));
    sycl::buffer<float, 1> bufferO(output.data(), sycl::range<1>(output.size()));

    q.submit([&](sycl::handler &cgh) {
        auto out = sycl::stream(10240, 7680, cgh);

        auto accessorI = bufferI.get_access<sycl::access::mode::read>(cgh);
        auto accessorO = bufferO.get_access<sycl::access::mode::write>(cgh);
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(wg), cgh);

        sycl::range<3> grid_size(dim2, dim1, dim0 / wg);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_reduction3>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                reduction3_kernel(item, accessorI, accessorO, sdata.get_pointer(), out);
        });
    }).wait();
}

/******************************************** reduction4 **************************************************/
inline void reduction4_kernel(const sycl::nd_item<3> &item,
                        const sycl::accessor<float, 1, sycl::access::mode::read> &g_idata,
                        const sycl::accessor<float, 1, sycl::access::mode::write> &g_odata,
                        float *sdata,
                        const sycl::stream &out) {
    unsigned int tid = item.get_local_id(2);
    unsigned int gid = item.get_group(2) * (item.get_local_range(2) * 2) + tid;
    sdata[tid] = g_idata[gid] + g_idata[gid + item.get_local_range(2)];
    item.barrier(sycl::access::fence_space::local_space);

    // do reduction in shared mem
    for (unsigned int s = item.get_local_range(2) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[item.get_group(2)] = sdata[0];
}

void reduction4(sycl::queue &q, const std::vector<float> &input, std::vector<float> &output) {
    sycl::buffer<float, 1> bufferI(input.data(), sycl::range<1>(input.size()));
    sycl::buffer<float, 1> bufferO(output.data(), sycl::range<1>(output.size()));

    q.submit([&](sycl::handler &cgh) {
        auto out = sycl::stream(10240, 7680, cgh);

        auto accessorI = bufferI.get_access<sycl::access::mode::read>(cgh);
        auto accessorO = bufferO.get_access<sycl::access::mode::write>(cgh);
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(wg), cgh);

        sycl::range<3> grid_size(dim2, dim1, dim0 / wg / 2);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_reduction4>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                reduction4_kernel(item, accessorI, accessorO, sdata.get_pointer(), out);
        });
    }).wait();
}

/******************************************** reduction5 **************************************************/
inline void warpReduce5(volatile float* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

inline void reduction5_kernel(const sycl::nd_item<3> &item,
                        const sycl::accessor<float, 1, sycl::access::mode::read> &g_idata,
                        const sycl::accessor<float, 1, sycl::access::mode::write> &g_odata,
                        float *sdata,
                        const sycl::stream &out) {
    unsigned int tid = item.get_local_id(2);
    unsigned int gid = item.get_group(2) * (item.get_local_range(2) * 2) + tid;
    sdata[tid] = g_idata[gid] + g_idata[gid + item.get_local_range(2)];
    item.barrier(sycl::access::fence_space::local_space);

    // do reduction in shared mem
    for (unsigned int s = item.get_local_range(2) / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    if (tid < 32) warpReduce5(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) g_odata[item.get_group(2)] = sdata[0];
}

void reduction5(sycl::queue &q, const std::vector<float> &input, std::vector<float> &output) {
    sycl::buffer<float, 1> bufferI(input.data(), sycl::range<1>(input.size()));
    sycl::buffer<float, 1> bufferO(output.data(), sycl::range<1>(output.size()));

    q.submit([&](sycl::handler &cgh) {
        auto out = sycl::stream(10240, 7680, cgh);

        auto accessorI = bufferI.get_access<sycl::access::mode::read>(cgh);
        auto accessorO = bufferO.get_access<sycl::access::mode::write>(cgh);
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(wg), cgh);

        sycl::range<3> grid_size(dim2, dim1, dim0 / wg / 2);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_reduction5>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                reduction5_kernel(item, accessorI, accessorO, sdata.get_pointer(), out);
        });
    }).wait();
}

/******************************************** reduction6 **************************************************/
template <unsigned int blockSize>
inline void warpReduce(volatile float* sdata, unsigned int tid) {
    if constexpr (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if constexpr (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if constexpr (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if constexpr (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if constexpr (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if constexpr (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
inline void reduction6_kernel(const sycl::nd_item<3> &item,
                        const sycl::accessor<float, 1, sycl::access::mode::read> &g_idata,
                        const sycl::accessor<float, 1, sycl::access::mode::write> &g_odata,
                        float *sdata,
                        const sycl::stream &out) {
    unsigned int tid = item.get_local_id(2);
    unsigned int gid = item.get_group(2) * (item.get_local_range(2) * 2) + tid;
    sdata[tid] = g_idata[gid] + g_idata[gid + item.get_local_range(2)];
    item.barrier(sycl::access::fence_space::local_space);

    // do reduction in shared mem
    for (unsigned int s = item.get_local_range(2) / 2; s > 512; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    if constexpr (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if constexpr (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if constexpr (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if constexpr (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) g_odata[item.get_group(2)] = sdata[0];
}

void reduction6(sycl::queue &q, const std::vector<float> &input, std::vector<float> &output) {
    sycl::buffer<float, 1> bufferI(input.data(), sycl::range<1>(input.size()));
    sycl::buffer<float, 1> bufferO(output.data(), sycl::range<1>(output.size()));

    q.submit([&](sycl::handler &cgh) {
        auto out = sycl::stream(10240, 7680, cgh);

        auto accessorI = bufferI.get_access<sycl::access::mode::read>(cgh);
        auto accessorO = bufferO.get_access<sycl::access::mode::write>(cgh);
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(wg), cgh);

        sycl::range<3> grid_size(dim2, dim1, dim0 / wg / 2);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_reduction6>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                switch (wg) {
                    case 1024:
                        reduction6_kernel<1024>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 512:
                        reduction6_kernel<512>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 256:
                        reduction6_kernel<256>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 128:
                        reduction6_kernel<128>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 64:
                        reduction6_kernel<64>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 32:
                        reduction6_kernel<32>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 16:
                        reduction6_kernel<16>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 8:
                        reduction6_kernel<8>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 4:
                        reduction6_kernel<4>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 2:
                        reduction6_kernel<2>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                    case 1:
                        reduction6_kernel<1>(item, accessorI, accessorO, sdata.get_pointer(), out); break;
                }
        });
    }).wait();
}

/******************************************** reduction7 **************************************************/
template <unsigned int blockSize>
inline void reduction7_kernel(const sycl::nd_item<3> &item,
                        const sycl::accessor<float, 1, sycl::access::mode::read> &g_idata,
                        const sycl::accessor<float, 1, sycl::access::mode::write> &g_odata,
                        unsigned int n,
                        float *sdata,
                        const sycl::stream &out) {
    unsigned int tid = item.get_local_id(2);
    unsigned int gid = item.get_group(2) * (item.get_local_range(2) * 2) + tid;
    unsigned int gridSize = blockSize * 2 * item.get_group_range(2);
    sdata[tid] = 0;
    while (gid < n) {
        sdata[tid] += g_idata[gid] + g_idata[gid + blockSize];
        gid += gridSize;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // do reduction in shared mem
    for (unsigned int s = item.get_local_range(2) / 2; s > 512; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    if constexpr (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if constexpr (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if constexpr (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if constexpr (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) g_odata[item.get_group(2)] = sdata[0];
}

void reduction7(sycl::queue &q, const std::vector<float> &input, std::vector<float> &output) {
    sycl::buffer<float, 1> bufferI(input.data(), sycl::range<1>(input.size()));
    sycl::buffer<float, 1> bufferO(output.data(), sycl::range<1>(output.size()));

    q.submit([&](sycl::handler &cgh) {
        auto out = sycl::stream(10240, 7680, cgh);

        auto accessorI = bufferI.get_access<sycl::access::mode::read>(cgh);
        auto accessorO = bufferO.get_access<sycl::access::mode::write>(cgh);
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(wg), cgh);

        sycl::range<3> grid_size(dim2, dim1, 1);
        sycl::range<3> work_group_size(1, 1, wg);

        cgh.parallel_for<class kernel_reduction7>(
            sycl::nd_range(grid_size * work_group_size, work_group_size),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                switch (wg) {
                    case 1024:
                        reduction7_kernel<1024>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 512:
                        reduction7_kernel<512>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 256:
                        reduction7_kernel<256>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 128:
                        reduction7_kernel<128>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 64:
                        reduction7_kernel<64>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 32:
                        reduction7_kernel<32>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 16:
                        reduction7_kernel<16>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 8:
                        reduction7_kernel<8>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 4:
                        reduction7_kernel<4>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 2:
                        reduction7_kernel<2>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                    case 1:
                        reduction7_kernel<1>(item, accessorI, accessorO, dim0, sdata.get_pointer(), out); break;
                }
        });
    }).wait();
}

int main() {
    sycl::queue q;
    std::vector<float> input(dim2 * dim1 * dim0, 0.0f);

    for (int i = 0; i < input.size(); ++i) {
        input[i] = 0.0001f * i;
    }

    {
        std::vector<float> output(dim0 / wg, 0.0f);
        reduction1(q, input, output);
        auto start_time = std::chrono::high_resolution_clock::now();
        reduction1(q, input, output);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("reduction1 latency: %.6f ms, ", during_ms);

        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            // std::cout << output[i] << ", ";
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        std::vector<float> output(dim0 / wg, 0.0f);
        reduction2(q, input, output);
        auto start_time = std::chrono::high_resolution_clock::now();
        reduction2(q, input, output);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("reduction2 latency: %.6f ms, ", during_ms);

        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            // std::cout << output[i] << ", ";
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        std::vector<float> output(dim0 / wg, 0.0f);
        reduction3(q, input, output);
        auto start_time = std::chrono::high_resolution_clock::now();
        reduction3(q, input, output);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("reduction3 latency: %.6f ms, ", during_ms);

        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            // std::cout << output[i] << ", ";
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        std::vector<float> output(dim0 / wg / 2, 0.0f);
        reduction4(q, input, output);
        auto start_time = std::chrono::high_resolution_clock::now();
        reduction4(q, input, output);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("reduction4 latency: %.6f ms, ", during_ms);

        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            // std::cout << output[i] << ", ";
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        std::vector<float> output(dim0 / wg / 2, 0.0f);
        reduction5(q, input, output);
        auto start_time = std::chrono::high_resolution_clock::now();
        reduction5(q, input, output);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("reduction5 latency: %.6f ms, ", during_ms);

        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            // std::cout << output[i] << ", ";
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        std::vector<float> output(dim0 / wg / 2, 0.0f);
        reduction6(q, input, output);
        auto start_time = std::chrono::high_resolution_clock::now();
        reduction6(q, input, output);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("reduction6 latency: %.6f ms, ", during_ms);

        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            // std::cout << output[i] << ", ";
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    {
        std::vector<float> output(1, 0.0f);
        reduction7(q, input, output);
        auto start_time = std::chrono::high_resolution_clock::now();
        reduction7(q, input, output);
        auto end_time = std::chrono::high_resolution_clock::now();
        float during_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        printf("reduction7 latency: %.6f ms, ", during_ms);

        std::cout << "gpu sum: ";
        float sum_gpu = 0.0f;
        for (int i = 0; i < output.size(); ++i) {
            // std::cout << output[i] << ", ";
            sum_gpu += output[i];
        }
        std::cout << "+= " << sum_gpu << std::endl;
    }

    float sum_cpu = 0.0f;
    for (int i = 0; i < input.size(); ++i) {
        sum_cpu += input[i];
    }
    std::cout << "cpu sum: " << sum_cpu << std::endl;

    return 0;
}
