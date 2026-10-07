#include <stdint.h>
#include <cuda_fp16.h>
// Adapted from ggml-cuda/mmq.cuh (load_tiles_q8_0,
// vec_dot_q8_0_q8_1_dp4a, mul_mat_q_process_tile).
// Copyright (c) 2023-2026 The ggml authors. MIT license: GGML-LICENSE.
// Weight input is this engine's aligned 36-byte Q8_0, not GGML's 34-byte block.
// Packing deliberately preserves the existing quantizer and accumulation order.
// Launch contract: pack=(32,1,1); GEMM=(32,8,1), tile=(64 rows,64 columns).
// K must be divisible by 256; NP rounds N up to 64. Each warp handles one
// column modulo 8, each lane one row modulo 32. The 65-int weight row stride
// and padded scale stride avoid the bank conflicts of the prior prototypes.

struct RefQ8 { float d[4]; int q[32]; };
static_assert(sizeof(RefQ8) == 144, "D4 block ABI");
extern "C" __global__ void la_q8_mmq_pack(const int8_t* qs,const float* sc,RefQ8* out,int K,int N,int NP){
 int c=blockIdx.x, kb=blockIdx.y, lane=threadIdx.x;
 RefQ8* p=out+(size_t)kb*NP+c;
 p->q[lane]=(c<N && kb*128+lane*4<K)?*(const int*)(qs+(size_t)c*K+kb*128+lane*4):0;
 if(lane<4)p->d[lane]=(c<N && kb*4+lane<K/32)?sc[(size_t)c*(K/32)+kb*4+lane]:0;
}
extern "C" __global__ void la_q8_mmq_ref(const uint8_t* w,const RefQ8* y,float* dst,int K,int M,int N,int NP){
 constexpr int MX=64;
 __shared__ int xq[64*65];
 __shared__ float xd[64*8+64/4];
 __shared__ RefQ8 yt[MX];
 int lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane;
 int rbase=blockIdx.x*64,cbase=blockIdx.y*MX,nb=K/32;
 float sum[MX/8][2]={};
 for(int kb=0;kb<nb;kb+=8){
  #pragma unroll
  for(int i0=0;i0<64;i0+=8){int i=i0+warp,rr=rbase+i;
   #pragma unroll
   for(int h=0;h<2;h++){int wi=lane+h*32,b=kb+wi/8;
    xq[i*65+wi]=(rr<M&&b<nb)?*(const int*)(w+((size_t)rr*nb+b)*36+4+(wi%8)*4):0;
   }
  }
  #pragma unroll
  for(int i0=0;i0<64;i0+=32){int i=i0+warp*4+lane/8,rr=rbase+i,b=kb+lane%8;
   xd[i*8+i/4+lane%8]=(rr<M&&b<nb)?__half2float(*(const __half*)(w+((size_t)rr*nb+b)*36)):0;
  }
  #pragma unroll
  for(int h=0;h<2;h++){
   const int* by=(const int*)(y+(size_t)(kb/4+h)*NP+cbase);
   int* ty=(int*)yt;
   for(int l=tid;l<MX*36;l+=256)ty[l]=by[l];
   __syncthreads();
   for(int k01=0;k01<32;k01+=8){int k0=h*32+k01;
    #pragma unroll
    for(int j0=0;j0<MX;j0+=8){int j=j0+warp;
     #pragma unroll
     for(int i0=0;i0<64;i0+=32){int i=i0+lane,dot=0;
      #pragma unroll
      for(int t=0;t<8;t++)dot=__dp4a(xq[i*65+k0+t],yt[j].q[k01+t],dot);
      sum[j0/8][i0/32]+=xd[i*8+i/4+k0/8]*yt[j].d[k01/8]*(float)dot;
     }
    }
   }
   __syncthreads();
  }
 }
 #pragma unroll
 for(int j0=0;j0<MX;j0+=8){int c=cbase+j0+warp;
  #pragma unroll
  for(int i0=0;i0<64;i0+=32){int r=rbase+i0+lane;if(c<N&&r<M)dst[(size_t)c*M+r]=sum[j0/8][i0/32];}
 }
}
