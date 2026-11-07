// MoE 槽位探针：把「本层有没有专家不在显存槽位」写成 pinned 里的一个计数。
//
// GPU 侧的调度需要一个「缺不缺专家」的信号，但这个信号不能由 CPU 同步去读
// （那就是每层一次 D2H，正好是我们想消灭的那个停顿）。做法是反过来：
// kernel 直接写主机内存（cudaHostAlloc+Mapped 的 pinned 页，GPU 可写），
// CPU 在下一次 forward 的开头读它 —— 那时上一次的 GPU 工作早已完成，读是免费的。
//
// 只写一个 int：miss 的个数。0 表示本层全部命中，CPU 可以整层不参与。
extern "C" __global__ void moe_probe_slots(
    const int* __restrict__ slot_ids,
    int n_exp,
    unsigned char* __restrict__ host_miss) {
    // 单 block 单 warp 就够：n_exp 是 6（最多 8）
    if (threadIdx.x == 0) {
        int miss = 0;
        for (int e = 0; e < n_exp; ++e) {
            if (slot_ids[e] < 0) ++miss;
        }
        // 写一个字节（pinned host 内存，GPU 直接可写）
        *reinterpret_cast<unsigned int*>(host_miss) = (unsigned int)miss;
    }
}
