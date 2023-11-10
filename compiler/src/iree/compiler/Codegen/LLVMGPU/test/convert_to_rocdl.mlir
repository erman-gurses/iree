// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-rocdl))))" %s | FileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @abs_ex_dispatch_0 {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @abs_ex_dispatch_0 layout(#pipeline_layout)
    builtin.module {
      func.func @abs_ex_dispatch_0() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) flags(ReadOnly) : memref<16xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %1[%7] : memref<16xf32>
        %10 = memref.load %2[%7] : memref<16xf32>
        %11 = arith.addf %9, %10 : f32
        memref.store %11, %0[%7] : memref<16xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @abs_ex_dispatch_0
//  CHECK-SAME: (%{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.readonly},
//  CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias})
//      CHECK:    rocdl.workgroup.dim.x
//      CHECK:    llvm.fadd


// -----
// Test that maximum and minum are converted to max and min on rocm
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @abs_ex_dispatch_0 {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @abs_ex_dispatch_0 layout(#pipeline_layout)
    builtin.module {
      func.func @reduction_maximum() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) :
            memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64x64xf32,
            strided<[4096, 64, 1], offset: ?>>
      %2 = vector.load %0[%c0, %c0, %c0] : memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<2xf32>
      %3 = vector.reduction <maximumf>, %2 : vector<2xf32> into f32
      %4 = vector.splat %3 : vector<2xf32>
      vector.store %4, %1[%c0, %c0, %c0] : memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<2xf32>
      return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @reduction_maximum
// CHECK:  llvm.intr.vector.reduce.fmax({{.*}})  : (vector<2xf32>) -> f32

// -----
// Test that gpu barriers be lowered to `s_waitcnt lgkmcnt(0)\0As_barrier` on rocm
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_dispatch_0 {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @matmul_dispatch_0_matmul_transpose_b_4096x4096x4096_f16xf16xf32 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_dispatch_0_matmul_transpose_b_4096x4096x4096_f16xf16xf32() {
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4x32xf16>
        %c-4096 = arith.constant -4096 : index
        %c-1 = arith.constant -1 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c4096 = arith.constant 4096 : index
        %cst_0 = arith.constant dense<0.000000e+00> : vector<4x4xf32>
        %c0 = arith.constant 0 : index
        %alloc = memref.alloc() : memref<128x40xf16, #gpu.address_space<workgroup>>
        %alloc_1 = memref.alloc() : memref<32x40xf16, #gpu.address_space<workgroup>>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<4096x4096xf16, #gpu.address_space<global>>
        memref.assume_alignment %0, 64 : memref<4096x4096xf16, #gpu.address_space<global>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<4096x4096xf16, #gpu.address_space<global>>
        memref.assume_alignment %1, 64 : memref<4096x4096xf16, #gpu.address_space<global>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<4096x4096xf32, #gpu.address_space<global>>
        memref.assume_alignment %2, 64 : memref<4096x4096xf32, #gpu.address_space<global>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %3 = gpu.thread_id  x
        %4 = gpu.thread_id  y
        %5 = arith.muli %4, %c4 : index
        %6 = arith.cmpi slt, %workgroup_id_x, %c0 : index
        %7 = arith.subi %c-1, %workgroup_id_x : index
        %8 = arith.select %6, %7, %workgroup_id_x : index
        %9 = arith.divsi %8, %c32 : index
        %10 = arith.subi %c-1, %9 : index
        %11 = arith.select %6, %10, %9 : index
        %12 = arith.muli %11, %c32 : index
        %13 = arith.addi %5, %12 : index
        %14 = arith.muli %workgroup_id_x, %c128 : index
        %15 = arith.muli %3, %c4 : index
        %16 = arith.addi %14, %15 : index
        %17 = arith.muli %11, %c-4096 : index
        %18 = arith.addi %16, %17 : index
        %19 = arith.muli %3, %c8 : index
        %20 = arith.addi %5, %c1 : index
        %21 = arith.addi %5, %c2 : index
        %22 = arith.addi %5, %c3 : index
        %23 = arith.addi %15, %c1 : index
        %24 = arith.addi %15, %c2 : index
        %25 = arith.addi %15, %c3 : index
        cf.br ^bb1(%c0, %cst_0 : index, vector<4x4xf32>)
      ^bb1(%26: index, %27: vector<4x4xf32>):  // 2 preds: ^bb0, ^bb19
        %28 = arith.cmpi slt, %26, %c4096 : index
        cf.cond_br %28, ^bb2, ^bb20
      ^bb2:  // pred: ^bb1
        gpu.barrier
        cf.br ^bb3(%4 : index)
      ^bb3(%29: index):  // 2 preds: ^bb2, ^bb10
        %30 = arith.cmpi slt, %29, %c32 : index
        cf.cond_br %30, ^bb4, ^bb11(%4 : index)
      ^bb4:  // pred: ^bb3
        %31 = arith.addi %29, %12 : index
        cf.br ^bb5(%19 : index)
      ^bb5(%32: index):  // 2 preds: ^bb4, ^bb9
        %33 = arith.cmpi slt, %32, %c32 : index
        cf.cond_br %33, ^bb6, ^bb10
      ^bb6:  // pred: ^bb5
        %34 = arith.addi %26, %32 : index
        cf.br ^bb7(%c0 : index)
      ^bb7(%35: index):  // 2 preds: ^bb6, ^bb8
        %36 = arith.cmpi slt, %35, %c8 : index
        cf.cond_br %36, ^bb8, ^bb9
      ^bb8:  // pred: ^bb7
        %37 = arith.addi %34, %35 : index
        %38 = memref.load %0[%31, %37] : memref<4096x4096xf16, #gpu.address_space<global>>
        %39 = arith.addi %32, %35 : index
        memref.store %38, %alloc_1[%29, %39] : memref<32x40xf16, #gpu.address_space<workgroup>>
        %40 = arith.addi %35, %c1 : index
        cf.br ^bb7(%40 : index)
      ^bb9:  // pred: ^bb7
        %41 = arith.addi %32, %c256 : index
        cf.br ^bb5(%41 : index)
      ^bb10:  // pred: ^bb5
        %42 = arith.addi %29, %c8 : index
        cf.br ^bb3(%42 : index)
      ^bb11(%43: index):  // 2 preds: ^bb3, ^bb18
        %44 = arith.cmpi slt, %43, %c128 : index
        cf.cond_br %44, ^bb12, ^bb19
      ^bb12:  // pred: ^bb11
        %45 = arith.addi %43, %14 : index
        %46 = arith.addi %45, %17 : index
        cf.br ^bb13(%19 : index)
      ^bb13(%47: index):  // 2 preds: ^bb12, ^bb17
        %48 = arith.cmpi slt, %47, %c32 : index
        cf.cond_br %48, ^bb14, ^bb18
      ^bb14:  // pred: ^bb13
        %49 = arith.addi %26, %47 : index
        cf.br ^bb15(%c0 : index)
      ^bb15(%50: index):  // 2 preds: ^bb14, ^bb16
        %51 = arith.cmpi slt, %50, %c8 : index
        cf.cond_br %51, ^bb16, ^bb17
      ^bb16:  // pred: ^bb15
        %52 = arith.addi %49, %50 : index
        %53 = memref.load %1[%46, %52] : memref<4096x4096xf16, #gpu.address_space<global>>
        %54 = arith.addi %47, %50 : index
        memref.store %53, %alloc[%43, %54] : memref<128x40xf16, #gpu.address_space<workgroup>>
        %55 = arith.addi %50, %c1 : index
        cf.br ^bb15(%55 : index)
      ^bb17:  // pred: ^bb15
        %56 = arith.addi %47, %c256 : index
        cf.br ^bb13(%56 : index)
      ^bb18:  // pred: ^bb13
        %57 = arith.addi %43, %c8 : index
        cf.br ^bb11(%57 : index)
      ^bb19:  // pred: ^bb11
        gpu.barrier
        %58 = vector.load %alloc_1[%5, %c0] : memref<32x40xf16, #gpu.address_space<workgroup>>, vector<32xf16>
        %59 = vector.insert %58, %cst [0] : vector<32xf16> into vector<4x32xf16>
        %60 = vector.load %alloc_1[%20, %c0] : memref<32x40xf16, #gpu.address_space<workgroup>>, vector<32xf16>
        %61 = vector.insert %60, %59 [1] : vector<32xf16> into vector<4x32xf16>
        %62 = vector.load %alloc_1[%21, %c0] : memref<32x40xf16, #gpu.address_space<workgroup>>, vector<32xf16>
        %63 = vector.insert %62, %61 [2] : vector<32xf16> into vector<4x32xf16>
        %64 = vector.load %alloc_1[%22, %c0] : memref<32x40xf16, #gpu.address_space<workgroup>>, vector<32xf16>
        %65 = vector.insert %64, %63 [3] : vector<32xf16> into vector<4x32xf16>
        %66 = vector.load %alloc[%15, %c0] : memref<128x40xf16, #gpu.address_space<workgroup>>, vector<32xf16>
        %67 = vector.insert %66, %cst [0] : vector<32xf16> into vector<4x32xf16>
        %68 = vector.load %alloc[%23, %c0] : memref<128x40xf16, #gpu.address_space<workgroup>>, vector<32xf16>
        %69 = vector.insert %68, %67 [1] : vector<32xf16> into vector<4x32xf16>
        %70 = vector.load %alloc[%24, %c0] : memref<128x40xf16, #gpu.address_space<workgroup>>, vector<32xf16>
        %71 = vector.insert %70, %69 [2] : vector<32xf16> into vector<4x32xf16>
        %72 = vector.load %alloc[%25, %c0] : memref<128x40xf16, #gpu.address_space<workgroup>>, vector<32xf16>
        %73 = vector.insert %72, %71 [3] : vector<32xf16> into vector<4x32xf16>
        %74 = vector.transpose %65, [1, 0] : vector<4x32xf16> to vector<32x4xf16>
        %75 = vector.transpose %73, [1, 0] : vector<4x32xf16> to vector<32x4xf16>
        %76 = vector.extract %74[0] : vector<4xf16> from vector<32x4xf16>
        %77 = vector.extract %75[0] : vector<4xf16> from vector<32x4xf16>
        %78 = arith.extf %76 : vector<4xf16> to vector<4xf32>
        %79 = arith.extf %77 : vector<4xf16> to vector<4xf32>
        %80 = vector.extract %78[0] : f32 from vector<4xf32>
        %81 = vector.splat %80 : vector<4xf32>
        %82 = vector.extract %27[0] : vector<4xf32> from vector<4x4xf32>
        %83 = vector.fma %81, %79, %82 : vector<4xf32>
        %84 = vector.extract %78[1] : f32 from vector<4xf32>
        %85 = vector.splat %84 : vector<4xf32>
        %86 = vector.extract %27[1] : vector<4xf32> from vector<4x4xf32>
        %87 = vector.fma %85, %79, %86 : vector<4xf32>
        %88 = vector.extract %78[2] : f32 from vector<4xf32>
        %89 = vector.splat %88 : vector<4xf32>
        %90 = vector.extract %27[2] : vector<4xf32> from vector<4x4xf32>
        %91 = vector.fma %89, %79, %90 : vector<4xf32>
        %92 = vector.extract %78[3] : f32 from vector<4xf32>
        %93 = vector.splat %92 : vector<4xf32>
        %94 = vector.extract %27[3] : vector<4xf32> from vector<4x4xf32>
        %95 = vector.fma %93, %79, %94 : vector<4xf32>
        %96 = vector.extract %74[1] : vector<4xf16> from vector<32x4xf16>
        %97 = vector.extract %75[1] : vector<4xf16> from vector<32x4xf16>
        %98 = arith.extf %96 : vector<4xf16> to vector<4xf32>
        %99 = arith.extf %97 : vector<4xf16> to vector<4xf32>
        %100 = vector.extract %98[0] : f32 from vector<4xf32>
        %101 = vector.splat %100 : vector<4xf32>
        %102 = vector.fma %101, %99, %83 : vector<4xf32>
        %103 = vector.extract %98[1] : f32 from vector<4xf32>
        %104 = vector.splat %103 : vector<4xf32>
        %105 = vector.fma %104, %99, %87 : vector<4xf32>
        %106 = vector.extract %98[2] : f32 from vector<4xf32>
        %107 = vector.splat %106 : vector<4xf32>
        %108 = vector.fma %107, %99, %91 : vector<4xf32>
        %109 = vector.extract %98[3] : f32 from vector<4xf32>
        %110 = vector.splat %109 : vector<4xf32>
        %111 = vector.fma %110, %99, %95 : vector<4xf32>
        %112 = vector.extract %74[2] : vector<4xf16> from vector<32x4xf16>
        %113 = vector.extract %75[2] : vector<4xf16> from vector<32x4xf16>
        %114 = arith.extf %112 : vector<4xf16> to vector<4xf32>
        %115 = arith.extf %113 : vector<4xf16> to vector<4xf32>
        %116 = vector.extract %114[0] : f32 from vector<4xf32>
        %117 = vector.splat %116 : vector<4xf32>
        %118 = vector.fma %117, %115, %102 : vector<4xf32>
        %119 = vector.extract %114[1] : f32 from vector<4xf32>
        %120 = vector.splat %119 : vector<4xf32>
        %121 = vector.fma %120, %115, %105 : vector<4xf32>
        %122 = vector.extract %114[2] : f32 from vector<4xf32>
        %123 = vector.splat %122 : vector<4xf32>
        %124 = vector.fma %123, %115, %108 : vector<4xf32>
        %125 = vector.extract %114[3] : f32 from vector<4xf32>
        %126 = vector.splat %125 : vector<4xf32>
        %127 = vector.fma %126, %115, %111 : vector<4xf32>
        %128 = vector.extract %74[3] : vector<4xf16> from vector<32x4xf16>
        %129 = vector.extract %75[3] : vector<4xf16> from vector<32x4xf16>
        %130 = arith.extf %128 : vector<4xf16> to vector<4xf32>
        %131 = arith.extf %129 : vector<4xf16> to vector<4xf32>
        %132 = vector.extract %130[0] : f32 from vector<4xf32>
        %133 = vector.splat %132 : vector<4xf32>
        %134 = vector.fma %133, %131, %118 : vector<4xf32>
        %135 = vector.extract %130[1] : f32 from vector<4xf32>
        %136 = vector.splat %135 : vector<4xf32>
        %137 = vector.fma %136, %131, %121 : vector<4xf32>
        %138 = vector.extract %130[2] : f32 from vector<4xf32>
        %139 = vector.splat %138 : vector<4xf32>
        %140 = vector.fma %139, %131, %124 : vector<4xf32>
        %141 = vector.extract %130[3] : f32 from vector<4xf32>
        %142 = vector.splat %141 : vector<4xf32>
        %143 = vector.fma %142, %131, %127 : vector<4xf32>
        %144 = vector.extract %74[4] : vector<4xf16> from vector<32x4xf16>
        %145 = vector.extract %75[4] : vector<4xf16> from vector<32x4xf16>
        %146 = arith.extf %144 : vector<4xf16> to vector<4xf32>
        %147 = arith.extf %145 : vector<4xf16> to vector<4xf32>
        %148 = vector.extract %146[0] : f32 from vector<4xf32>
        %149 = vector.splat %148 : vector<4xf32>
        %150 = vector.fma %149, %147, %134 : vector<4xf32>
        %151 = vector.extract %146[1] : f32 from vector<4xf32>
        %152 = vector.splat %151 : vector<4xf32>
        %153 = vector.fma %152, %147, %137 : vector<4xf32>
        %154 = vector.extract %146[2] : f32 from vector<4xf32>
        %155 = vector.splat %154 : vector<4xf32>
        %156 = vector.fma %155, %147, %140 : vector<4xf32>
        %157 = vector.extract %146[3] : f32 from vector<4xf32>
        %158 = vector.splat %157 : vector<4xf32>
        %159 = vector.fma %158, %147, %143 : vector<4xf32>
        %160 = vector.extract %74[5] : vector<4xf16> from vector<32x4xf16>
        %161 = vector.extract %75[5] : vector<4xf16> from vector<32x4xf16>
        %162 = arith.extf %160 : vector<4xf16> to vector<4xf32>
        %163 = arith.extf %161 : vector<4xf16> to vector<4xf32>
        %164 = vector.extract %162[0] : f32 from vector<4xf32>
        %165 = vector.splat %164 : vector<4xf32>
        %166 = vector.fma %165, %163, %150 : vector<4xf32>
        %167 = vector.extract %162[1] : f32 from vector<4xf32>
        %168 = vector.splat %167 : vector<4xf32>
        %169 = vector.fma %168, %163, %153 : vector<4xf32>
        %170 = vector.extract %162[2] : f32 from vector<4xf32>
        %171 = vector.splat %170 : vector<4xf32>
        %172 = vector.fma %171, %163, %156 : vector<4xf32>
        %173 = vector.extract %162[3] : f32 from vector<4xf32>
        %174 = vector.splat %173 : vector<4xf32>
        %175 = vector.fma %174, %163, %159 : vector<4xf32>
        %176 = vector.extract %74[6] : vector<4xf16> from vector<32x4xf16>
        %177 = vector.extract %75[6] : vector<4xf16> from vector<32x4xf16>
        %178 = arith.extf %176 : vector<4xf16> to vector<4xf32>
        %179 = arith.extf %177 : vector<4xf16> to vector<4xf32>
        %180 = vector.extract %178[0] : f32 from vector<4xf32>
        %181 = vector.splat %180 : vector<4xf32>
        %182 = vector.fma %181, %179, %166 : vector<4xf32>
        %183 = vector.extract %178[1] : f32 from vector<4xf32>
        %184 = vector.splat %183 : vector<4xf32>
        %185 = vector.fma %184, %179, %169 : vector<4xf32>
        %186 = vector.extract %178[2] : f32 from vector<4xf32>
        %187 = vector.splat %186 : vector<4xf32>
        %188 = vector.fma %187, %179, %172 : vector<4xf32>
        %189 = vector.extract %178[3] : f32 from vector<4xf32>
        %190 = vector.splat %189 : vector<4xf32>
        %191 = vector.fma %190, %179, %175 : vector<4xf32>
        %192 = vector.extract %74[7] : vector<4xf16> from vector<32x4xf16>
        %193 = vector.extract %75[7] : vector<4xf16> from vector<32x4xf16>
        %194 = arith.extf %192 : vector<4xf16> to vector<4xf32>
        %195 = arith.extf %193 : vector<4xf16> to vector<4xf32>
        %196 = vector.extract %194[0] : f32 from vector<4xf32>
        %197 = vector.splat %196 : vector<4xf32>
        %198 = vector.fma %197, %195, %182 : vector<4xf32>
        %199 = vector.extract %194[1] : f32 from vector<4xf32>
        %200 = vector.splat %199 : vector<4xf32>
        %201 = vector.fma %200, %195, %185 : vector<4xf32>
        %202 = vector.extract %194[2] : f32 from vector<4xf32>
        %203 = vector.splat %202 : vector<4xf32>
        %204 = vector.fma %203, %195, %188 : vector<4xf32>
        %205 = vector.extract %194[3] : f32 from vector<4xf32>
        %206 = vector.splat %205 : vector<4xf32>
        %207 = vector.fma %206, %195, %191 : vector<4xf32>
        %208 = vector.extract %74[8] : vector<4xf16> from vector<32x4xf16>
        %209 = vector.extract %75[8] : vector<4xf16> from vector<32x4xf16>
        %210 = arith.extf %208 : vector<4xf16> to vector<4xf32>
        %211 = arith.extf %209 : vector<4xf16> to vector<4xf32>
        %212 = vector.extract %210[0] : f32 from vector<4xf32>
        %213 = vector.splat %212 : vector<4xf32>
        %214 = vector.fma %213, %211, %198 : vector<4xf32>
        %215 = vector.extract %210[1] : f32 from vector<4xf32>
        %216 = vector.splat %215 : vector<4xf32>
        %217 = vector.fma %216, %211, %201 : vector<4xf32>
        %218 = vector.extract %210[2] : f32 from vector<4xf32>
        %219 = vector.splat %218 : vector<4xf32>
        %220 = vector.fma %219, %211, %204 : vector<4xf32>
        %221 = vector.extract %210[3] : f32 from vector<4xf32>
        %222 = vector.splat %221 : vector<4xf32>
        %223 = vector.fma %222, %211, %207 : vector<4xf32>
        %224 = vector.extract %74[9] : vector<4xf16> from vector<32x4xf16>
        %225 = vector.extract %75[9] : vector<4xf16> from vector<32x4xf16>
        %226 = arith.extf %224 : vector<4xf16> to vector<4xf32>
        %227 = arith.extf %225 : vector<4xf16> to vector<4xf32>
        %228 = vector.extract %226[0] : f32 from vector<4xf32>
        %229 = vector.splat %228 : vector<4xf32>
        %230 = vector.fma %229, %227, %214 : vector<4xf32>
        %231 = vector.extract %226[1] : f32 from vector<4xf32>
        %232 = vector.splat %231 : vector<4xf32>
        %233 = vector.fma %232, %227, %217 : vector<4xf32>
        %234 = vector.extract %226[2] : f32 from vector<4xf32>
        %235 = vector.splat %234 : vector<4xf32>
        %236 = vector.fma %235, %227, %220 : vector<4xf32>
        %237 = vector.extract %226[3] : f32 from vector<4xf32>
        %238 = vector.splat %237 : vector<4xf32>
        %239 = vector.fma %238, %227, %223 : vector<4xf32>
        %240 = vector.extract %74[10] : vector<4xf16> from vector<32x4xf16>
        %241 = vector.extract %75[10] : vector<4xf16> from vector<32x4xf16>
        %242 = arith.extf %240 : vector<4xf16> to vector<4xf32>
        %243 = arith.extf %241 : vector<4xf16> to vector<4xf32>
        %244 = vector.extract %242[0] : f32 from vector<4xf32>
        %245 = vector.splat %244 : vector<4xf32>
        %246 = vector.fma %245, %243, %230 : vector<4xf32>
        %247 = vector.extract %242[1] : f32 from vector<4xf32>
        %248 = vector.splat %247 : vector<4xf32>
        %249 = vector.fma %248, %243, %233 : vector<4xf32>
        %250 = vector.extract %242[2] : f32 from vector<4xf32>
        %251 = vector.splat %250 : vector<4xf32>
        %252 = vector.fma %251, %243, %236 : vector<4xf32>
        %253 = vector.extract %242[3] : f32 from vector<4xf32>
        %254 = vector.splat %253 : vector<4xf32>
        %255 = vector.fma %254, %243, %239 : vector<4xf32>
        %256 = vector.extract %74[11] : vector<4xf16> from vector<32x4xf16>
        %257 = vector.extract %75[11] : vector<4xf16> from vector<32x4xf16>
        %258 = arith.extf %256 : vector<4xf16> to vector<4xf32>
        %259 = arith.extf %257 : vector<4xf16> to vector<4xf32>
        %260 = vector.extract %258[0] : f32 from vector<4xf32>
        %261 = vector.splat %260 : vector<4xf32>
        %262 = vector.fma %261, %259, %246 : vector<4xf32>
        %263 = vector.extract %258[1] : f32 from vector<4xf32>
        %264 = vector.splat %263 : vector<4xf32>
        %265 = vector.fma %264, %259, %249 : vector<4xf32>
        %266 = vector.extract %258[2] : f32 from vector<4xf32>
        %267 = vector.splat %266 : vector<4xf32>
        %268 = vector.fma %267, %259, %252 : vector<4xf32>
        %269 = vector.extract %258[3] : f32 from vector<4xf32>
        %270 = vector.splat %269 : vector<4xf32>
        %271 = vector.fma %270, %259, %255 : vector<4xf32>
        %272 = vector.extract %74[12] : vector<4xf16> from vector<32x4xf16>
        %273 = vector.extract %75[12] : vector<4xf16> from vector<32x4xf16>
        %274 = arith.extf %272 : vector<4xf16> to vector<4xf32>
        %275 = arith.extf %273 : vector<4xf16> to vector<4xf32>
        %276 = vector.extract %274[0] : f32 from vector<4xf32>
        %277 = vector.splat %276 : vector<4xf32>
        %278 = vector.fma %277, %275, %262 : vector<4xf32>
        %279 = vector.extract %274[1] : f32 from vector<4xf32>
        %280 = vector.splat %279 : vector<4xf32>
        %281 = vector.fma %280, %275, %265 : vector<4xf32>
        %282 = vector.extract %274[2] : f32 from vector<4xf32>
        %283 = vector.splat %282 : vector<4xf32>
        %284 = vector.fma %283, %275, %268 : vector<4xf32>
        %285 = vector.extract %274[3] : f32 from vector<4xf32>
        %286 = vector.splat %285 : vector<4xf32>
        %287 = vector.fma %286, %275, %271 : vector<4xf32>
        %288 = vector.extract %74[13] : vector<4xf16> from vector<32x4xf16>
        %289 = vector.extract %75[13] : vector<4xf16> from vector<32x4xf16>
        %290 = arith.extf %288 : vector<4xf16> to vector<4xf32>
        %291 = arith.extf %289 : vector<4xf16> to vector<4xf32>
        %292 = vector.extract %290[0] : f32 from vector<4xf32>
        %293 = vector.splat %292 : vector<4xf32>
        %294 = vector.fma %293, %291, %278 : vector<4xf32>
        %295 = vector.extract %290[1] : f32 from vector<4xf32>
        %296 = vector.splat %295 : vector<4xf32>
        %297 = vector.fma %296, %291, %281 : vector<4xf32>
        %298 = vector.extract %290[2] : f32 from vector<4xf32>
        %299 = vector.splat %298 : vector<4xf32>
        %300 = vector.fma %299, %291, %284 : vector<4xf32>
        %301 = vector.extract %290[3] : f32 from vector<4xf32>
        %302 = vector.splat %301 : vector<4xf32>
        %303 = vector.fma %302, %291, %287 : vector<4xf32>
        %304 = vector.extract %74[14] : vector<4xf16> from vector<32x4xf16>
        %305 = vector.extract %75[14] : vector<4xf16> from vector<32x4xf16>
        %306 = arith.extf %304 : vector<4xf16> to vector<4xf32>
        %307 = arith.extf %305 : vector<4xf16> to vector<4xf32>
        %308 = vector.extract %306[0] : f32 from vector<4xf32>
        %309 = vector.splat %308 : vector<4xf32>
        %310 = vector.fma %309, %307, %294 : vector<4xf32>
        %311 = vector.extract %306[1] : f32 from vector<4xf32>
        %312 = vector.splat %311 : vector<4xf32>
        %313 = vector.fma %312, %307, %297 : vector<4xf32>
        %314 = vector.extract %306[2] : f32 from vector<4xf32>
        %315 = vector.splat %314 : vector<4xf32>
        %316 = vector.fma %315, %307, %300 : vector<4xf32>
        %317 = vector.extract %306[3] : f32 from vector<4xf32>
        %318 = vector.splat %317 : vector<4xf32>
        %319 = vector.fma %318, %307, %303 : vector<4xf32>
        %320 = vector.extract %74[15] : vector<4xf16> from vector<32x4xf16>
        %321 = vector.extract %75[15] : vector<4xf16> from vector<32x4xf16>
        %322 = arith.extf %320 : vector<4xf16> to vector<4xf32>
        %323 = arith.extf %321 : vector<4xf16> to vector<4xf32>
        %324 = vector.extract %322[0] : f32 from vector<4xf32>
        %325 = vector.splat %324 : vector<4xf32>
        %326 = vector.fma %325, %323, %310 : vector<4xf32>
        %327 = vector.extract %322[1] : f32 from vector<4xf32>
        %328 = vector.splat %327 : vector<4xf32>
        %329 = vector.fma %328, %323, %313 : vector<4xf32>
        %330 = vector.extract %322[2] : f32 from vector<4xf32>
        %331 = vector.splat %330 : vector<4xf32>
        %332 = vector.fma %331, %323, %316 : vector<4xf32>
        %333 = vector.extract %322[3] : f32 from vector<4xf32>
        %334 = vector.splat %333 : vector<4xf32>
        %335 = vector.fma %334, %323, %319 : vector<4xf32>
        %336 = vector.extract %74[16] : vector<4xf16> from vector<32x4xf16>
        %337 = vector.extract %75[16] : vector<4xf16> from vector<32x4xf16>
        %338 = arith.extf %336 : vector<4xf16> to vector<4xf32>
        %339 = arith.extf %337 : vector<4xf16> to vector<4xf32>
        %340 = vector.extract %338[0] : f32 from vector<4xf32>
        %341 = vector.splat %340 : vector<4xf32>
        %342 = vector.fma %341, %339, %326 : vector<4xf32>
        %343 = vector.extract %338[1] : f32 from vector<4xf32>
        %344 = vector.splat %343 : vector<4xf32>
        %345 = vector.fma %344, %339, %329 : vector<4xf32>
        %346 = vector.extract %338[2] : f32 from vector<4xf32>
        %347 = vector.splat %346 : vector<4xf32>
        %348 = vector.fma %347, %339, %332 : vector<4xf32>
        %349 = vector.extract %338[3] : f32 from vector<4xf32>
        %350 = vector.splat %349 : vector<4xf32>
        %351 = vector.fma %350, %339, %335 : vector<4xf32>
        %352 = vector.extract %74[17] : vector<4xf16> from vector<32x4xf16>
        %353 = vector.extract %75[17] : vector<4xf16> from vector<32x4xf16>
        %354 = arith.extf %352 : vector<4xf16> to vector<4xf32>
        %355 = arith.extf %353 : vector<4xf16> to vector<4xf32>
        %356 = vector.extract %354[0] : f32 from vector<4xf32>
        %357 = vector.splat %356 : vector<4xf32>
        %358 = vector.fma %357, %355, %342 : vector<4xf32>
        %359 = vector.extract %354[1] : f32 from vector<4xf32>
        %360 = vector.splat %359 : vector<4xf32>
        %361 = vector.fma %360, %355, %345 : vector<4xf32>
        %362 = vector.extract %354[2] : f32 from vector<4xf32>
        %363 = vector.splat %362 : vector<4xf32>
        %364 = vector.fma %363, %355, %348 : vector<4xf32>
        %365 = vector.extract %354[3] : f32 from vector<4xf32>
        %366 = vector.splat %365 : vector<4xf32>
        %367 = vector.fma %366, %355, %351 : vector<4xf32>
        %368 = vector.extract %74[18] : vector<4xf16> from vector<32x4xf16>
        %369 = vector.extract %75[18] : vector<4xf16> from vector<32x4xf16>
        %370 = arith.extf %368 : vector<4xf16> to vector<4xf32>
        %371 = arith.extf %369 : vector<4xf16> to vector<4xf32>
        %372 = vector.extract %370[0] : f32 from vector<4xf32>
        %373 = vector.splat %372 : vector<4xf32>
        %374 = vector.fma %373, %371, %358 : vector<4xf32>
        %375 = vector.extract %370[1] : f32 from vector<4xf32>
        %376 = vector.splat %375 : vector<4xf32>
        %377 = vector.fma %376, %371, %361 : vector<4xf32>
        %378 = vector.extract %370[2] : f32 from vector<4xf32>
        %379 = vector.splat %378 : vector<4xf32>
        %380 = vector.fma %379, %371, %364 : vector<4xf32>
        %381 = vector.extract %370[3] : f32 from vector<4xf32>
        %382 = vector.splat %381 : vector<4xf32>
        %383 = vector.fma %382, %371, %367 : vector<4xf32>
        %384 = vector.extract %74[19] : vector<4xf16> from vector<32x4xf16>
        %385 = vector.extract %75[19] : vector<4xf16> from vector<32x4xf16>
        %386 = arith.extf %384 : vector<4xf16> to vector<4xf32>
        %387 = arith.extf %385 : vector<4xf16> to vector<4xf32>
        %388 = vector.extract %386[0] : f32 from vector<4xf32>
        %389 = vector.splat %388 : vector<4xf32>
        %390 = vector.fma %389, %387, %374 : vector<4xf32>
        %391 = vector.extract %386[1] : f32 from vector<4xf32>
        %392 = vector.splat %391 : vector<4xf32>
        %393 = vector.fma %392, %387, %377 : vector<4xf32>
        %394 = vector.extract %386[2] : f32 from vector<4xf32>
        %395 = vector.splat %394 : vector<4xf32>
        %396 = vector.fma %395, %387, %380 : vector<4xf32>
        %397 = vector.extract %386[3] : f32 from vector<4xf32>
        %398 = vector.splat %397 : vector<4xf32>
        %399 = vector.fma %398, %387, %383 : vector<4xf32>
        %400 = vector.extract %74[20] : vector<4xf16> from vector<32x4xf16>
        %401 = vector.extract %75[20] : vector<4xf16> from vector<32x4xf16>
        %402 = arith.extf %400 : vector<4xf16> to vector<4xf32>
        %403 = arith.extf %401 : vector<4xf16> to vector<4xf32>
        %404 = vector.extract %402[0] : f32 from vector<4xf32>
        %405 = vector.splat %404 : vector<4xf32>
        %406 = vector.fma %405, %403, %390 : vector<4xf32>
        %407 = vector.extract %402[1] : f32 from vector<4xf32>
        %408 = vector.splat %407 : vector<4xf32>
        %409 = vector.fma %408, %403, %393 : vector<4xf32>
        %410 = vector.extract %402[2] : f32 from vector<4xf32>
        %411 = vector.splat %410 : vector<4xf32>
        %412 = vector.fma %411, %403, %396 : vector<4xf32>
        %413 = vector.extract %402[3] : f32 from vector<4xf32>
        %414 = vector.splat %413 : vector<4xf32>
        %415 = vector.fma %414, %403, %399 : vector<4xf32>
        %416 = vector.extract %74[21] : vector<4xf16> from vector<32x4xf16>
        %417 = vector.extract %75[21] : vector<4xf16> from vector<32x4xf16>
        %418 = arith.extf %416 : vector<4xf16> to vector<4xf32>
        %419 = arith.extf %417 : vector<4xf16> to vector<4xf32>
        %420 = vector.extract %418[0] : f32 from vector<4xf32>
        %421 = vector.splat %420 : vector<4xf32>
        %422 = vector.fma %421, %419, %406 : vector<4xf32>
        %423 = vector.extract %418[1] : f32 from vector<4xf32>
        %424 = vector.splat %423 : vector<4xf32>
        %425 = vector.fma %424, %419, %409 : vector<4xf32>
        %426 = vector.extract %418[2] : f32 from vector<4xf32>
        %427 = vector.splat %426 : vector<4xf32>
        %428 = vector.fma %427, %419, %412 : vector<4xf32>
        %429 = vector.extract %418[3] : f32 from vector<4xf32>
        %430 = vector.splat %429 : vector<4xf32>
        %431 = vector.fma %430, %419, %415 : vector<4xf32>
        %432 = vector.extract %74[22] : vector<4xf16> from vector<32x4xf16>
        %433 = vector.extract %75[22] : vector<4xf16> from vector<32x4xf16>
        %434 = arith.extf %432 : vector<4xf16> to vector<4xf32>
        %435 = arith.extf %433 : vector<4xf16> to vector<4xf32>
        %436 = vector.extract %434[0] : f32 from vector<4xf32>
        %437 = vector.splat %436 : vector<4xf32>
        %438 = vector.fma %437, %435, %422 : vector<4xf32>
        %439 = vector.extract %434[1] : f32 from vector<4xf32>
        %440 = vector.splat %439 : vector<4xf32>
        %441 = vector.fma %440, %435, %425 : vector<4xf32>
        %442 = vector.extract %434[2] : f32 from vector<4xf32>
        %443 = vector.splat %442 : vector<4xf32>
        %444 = vector.fma %443, %435, %428 : vector<4xf32>
        %445 = vector.extract %434[3] : f32 from vector<4xf32>
        %446 = vector.splat %445 : vector<4xf32>
        %447 = vector.fma %446, %435, %431 : vector<4xf32>
        %448 = vector.extract %74[23] : vector<4xf16> from vector<32x4xf16>
        %449 = vector.extract %75[23] : vector<4xf16> from vector<32x4xf16>
        %450 = arith.extf %448 : vector<4xf16> to vector<4xf32>
        %451 = arith.extf %449 : vector<4xf16> to vector<4xf32>
        %452 = vector.extract %450[0] : f32 from vector<4xf32>
        %453 = vector.splat %452 : vector<4xf32>
        %454 = vector.fma %453, %451, %438 : vector<4xf32>
        %455 = vector.extract %450[1] : f32 from vector<4xf32>
        %456 = vector.splat %455 : vector<4xf32>
        %457 = vector.fma %456, %451, %441 : vector<4xf32>
        %458 = vector.extract %450[2] : f32 from vector<4xf32>
        %459 = vector.splat %458 : vector<4xf32>
        %460 = vector.fma %459, %451, %444 : vector<4xf32>
        %461 = vector.extract %450[3] : f32 from vector<4xf32>
        %462 = vector.splat %461 : vector<4xf32>
        %463 = vector.fma %462, %451, %447 : vector<4xf32>
        %464 = vector.extract %74[24] : vector<4xf16> from vector<32x4xf16>
        %465 = vector.extract %75[24] : vector<4xf16> from vector<32x4xf16>
        %466 = arith.extf %464 : vector<4xf16> to vector<4xf32>
        %467 = arith.extf %465 : vector<4xf16> to vector<4xf32>
        %468 = vector.extract %466[0] : f32 from vector<4xf32>
        %469 = vector.splat %468 : vector<4xf32>
        %470 = vector.fma %469, %467, %454 : vector<4xf32>
        %471 = vector.extract %466[1] : f32 from vector<4xf32>
        %472 = vector.splat %471 : vector<4xf32>
        %473 = vector.fma %472, %467, %457 : vector<4xf32>
        %474 = vector.extract %466[2] : f32 from vector<4xf32>
        %475 = vector.splat %474 : vector<4xf32>
        %476 = vector.fma %475, %467, %460 : vector<4xf32>
        %477 = vector.extract %466[3] : f32 from vector<4xf32>
        %478 = vector.splat %477 : vector<4xf32>
        %479 = vector.fma %478, %467, %463 : vector<4xf32>
        %480 = vector.extract %74[25] : vector<4xf16> from vector<32x4xf16>
        %481 = vector.extract %75[25] : vector<4xf16> from vector<32x4xf16>
        %482 = arith.extf %480 : vector<4xf16> to vector<4xf32>
        %483 = arith.extf %481 : vector<4xf16> to vector<4xf32>
        %484 = vector.extract %482[0] : f32 from vector<4xf32>
        %485 = vector.splat %484 : vector<4xf32>
        %486 = vector.fma %485, %483, %470 : vector<4xf32>
        %487 = vector.extract %482[1] : f32 from vector<4xf32>
        %488 = vector.splat %487 : vector<4xf32>
        %489 = vector.fma %488, %483, %473 : vector<4xf32>
        %490 = vector.extract %482[2] : f32 from vector<4xf32>
        %491 = vector.splat %490 : vector<4xf32>
        %492 = vector.fma %491, %483, %476 : vector<4xf32>
        %493 = vector.extract %482[3] : f32 from vector<4xf32>
        %494 = vector.splat %493 : vector<4xf32>
        %495 = vector.fma %494, %483, %479 : vector<4xf32>
        %496 = vector.extract %74[26] : vector<4xf16> from vector<32x4xf16>
        %497 = vector.extract %75[26] : vector<4xf16> from vector<32x4xf16>
        %498 = arith.extf %496 : vector<4xf16> to vector<4xf32>
        %499 = arith.extf %497 : vector<4xf16> to vector<4xf32>
        %500 = vector.extract %498[0] : f32 from vector<4xf32>
        %501 = vector.splat %500 : vector<4xf32>
        %502 = vector.fma %501, %499, %486 : vector<4xf32>
        %503 = vector.extract %498[1] : f32 from vector<4xf32>
        %504 = vector.splat %503 : vector<4xf32>
        %505 = vector.fma %504, %499, %489 : vector<4xf32>
        %506 = vector.extract %498[2] : f32 from vector<4xf32>
        %507 = vector.splat %506 : vector<4xf32>
        %508 = vector.fma %507, %499, %492 : vector<4xf32>
        %509 = vector.extract %498[3] : f32 from vector<4xf32>
        %510 = vector.splat %509 : vector<4xf32>
        %511 = vector.fma %510, %499, %495 : vector<4xf32>
        %512 = vector.extract %74[27] : vector<4xf16> from vector<32x4xf16>
        %513 = vector.extract %75[27] : vector<4xf16> from vector<32x4xf16>
        %514 = arith.extf %512 : vector<4xf16> to vector<4xf32>
        %515 = arith.extf %513 : vector<4xf16> to vector<4xf32>
        %516 = vector.extract %514[0] : f32 from vector<4xf32>
        %517 = vector.splat %516 : vector<4xf32>
        %518 = vector.fma %517, %515, %502 : vector<4xf32>
        %519 = vector.extract %514[1] : f32 from vector<4xf32>
        %520 = vector.splat %519 : vector<4xf32>
        %521 = vector.fma %520, %515, %505 : vector<4xf32>
        %522 = vector.extract %514[2] : f32 from vector<4xf32>
        %523 = vector.splat %522 : vector<4xf32>
        %524 = vector.fma %523, %515, %508 : vector<4xf32>
        %525 = vector.extract %514[3] : f32 from vector<4xf32>
        %526 = vector.splat %525 : vector<4xf32>
        %527 = vector.fma %526, %515, %511 : vector<4xf32>
        %528 = vector.extract %74[28] : vector<4xf16> from vector<32x4xf16>
        %529 = vector.extract %75[28] : vector<4xf16> from vector<32x4xf16>
        %530 = arith.extf %528 : vector<4xf16> to vector<4xf32>
        %531 = arith.extf %529 : vector<4xf16> to vector<4xf32>
        %532 = vector.extract %530[0] : f32 from vector<4xf32>
        %533 = vector.splat %532 : vector<4xf32>
        %534 = vector.fma %533, %531, %518 : vector<4xf32>
        %535 = vector.extract %530[1] : f32 from vector<4xf32>
        %536 = vector.splat %535 : vector<4xf32>
        %537 = vector.fma %536, %531, %521 : vector<4xf32>
        %538 = vector.extract %530[2] : f32 from vector<4xf32>
        %539 = vector.splat %538 : vector<4xf32>
        %540 = vector.fma %539, %531, %524 : vector<4xf32>
        %541 = vector.extract %530[3] : f32 from vector<4xf32>
        %542 = vector.splat %541 : vector<4xf32>
        %543 = vector.fma %542, %531, %527 : vector<4xf32>
        %544 = vector.extract %74[29] : vector<4xf16> from vector<32x4xf16>
        %545 = vector.extract %75[29] : vector<4xf16> from vector<32x4xf16>
        %546 = arith.extf %544 : vector<4xf16> to vector<4xf32>
        %547 = arith.extf %545 : vector<4xf16> to vector<4xf32>
        %548 = vector.extract %546[0] : f32 from vector<4xf32>
        %549 = vector.splat %548 : vector<4xf32>
        %550 = vector.fma %549, %547, %534 : vector<4xf32>
        %551 = vector.extract %546[1] : f32 from vector<4xf32>
        %552 = vector.splat %551 : vector<4xf32>
        %553 = vector.fma %552, %547, %537 : vector<4xf32>
        %554 = vector.extract %546[2] : f32 from vector<4xf32>
        %555 = vector.splat %554 : vector<4xf32>
        %556 = vector.fma %555, %547, %540 : vector<4xf32>
        %557 = vector.extract %546[3] : f32 from vector<4xf32>
        %558 = vector.splat %557 : vector<4xf32>
        %559 = vector.fma %558, %547, %543 : vector<4xf32>
        %560 = vector.extract %74[30] : vector<4xf16> from vector<32x4xf16>
        %561 = vector.extract %75[30] : vector<4xf16> from vector<32x4xf16>
        %562 = arith.extf %560 : vector<4xf16> to vector<4xf32>
        %563 = arith.extf %561 : vector<4xf16> to vector<4xf32>
        %564 = vector.extract %562[0] : f32 from vector<4xf32>
        %565 = vector.splat %564 : vector<4xf32>
        %566 = vector.fma %565, %563, %550 : vector<4xf32>
        %567 = vector.extract %562[1] : f32 from vector<4xf32>
        %568 = vector.splat %567 : vector<4xf32>
        %569 = vector.fma %568, %563, %553 : vector<4xf32>
        %570 = vector.extract %562[2] : f32 from vector<4xf32>
        %571 = vector.splat %570 : vector<4xf32>
        %572 = vector.fma %571, %563, %556 : vector<4xf32>
        %573 = vector.extract %562[3] : f32 from vector<4xf32>
        %574 = vector.splat %573 : vector<4xf32>
        %575 = vector.fma %574, %563, %559 : vector<4xf32>
        %576 = vector.extract %74[31] : vector<4xf16> from vector<32x4xf16>
        %577 = vector.extract %75[31] : vector<4xf16> from vector<32x4xf16>
        %578 = arith.extf %576 : vector<4xf16> to vector<4xf32>
        %579 = arith.extf %577 : vector<4xf16> to vector<4xf32>
        %580 = vector.extract %578[0] : f32 from vector<4xf32>
        %581 = vector.splat %580 : vector<4xf32>
        %582 = vector.fma %581, %579, %566 : vector<4xf32>
        %583 = vector.insert %582, %cst_0 [0] : vector<4xf32> into vector<4x4xf32>
        %584 = vector.extract %578[1] : f32 from vector<4xf32>
        %585 = vector.splat %584 : vector<4xf32>
        %586 = vector.fma %585, %579, %569 : vector<4xf32>
        %587 = vector.insert %586, %583 [1] : vector<4xf32> into vector<4x4xf32>
        %588 = vector.extract %578[2] : f32 from vector<4xf32>
        %589 = vector.splat %588 : vector<4xf32>
        %590 = vector.fma %589, %579, %572 : vector<4xf32>
        %591 = vector.insert %590, %587 [2] : vector<4xf32> into vector<4x4xf32>
        %592 = vector.extract %578[3] : f32 from vector<4xf32>
        %593 = vector.splat %592 : vector<4xf32>
        %594 = vector.fma %593, %579, %575 : vector<4xf32>
        %595 = vector.insert %594, %591 [3] : vector<4xf32> into vector<4x4xf32>
        %596 = arith.addi %26, %c32 : index
        cf.br ^bb1(%596, %595 : index, vector<4x4xf32>)
      ^bb20:  // pred: ^bb1
        %597 = vector.extract %27[0] : vector<4xf32> from vector<4x4xf32>
        vector.store %597, %2[%13, %18] : memref<4096x4096xf32, #gpu.address_space<global>>, vector<4xf32>
        %598 = arith.addi %13, %c1 : index
        %599 = vector.extract %27[1] : vector<4xf32> from vector<4x4xf32>
        vector.store %599, %2[%598, %18] : memref<4096x4096xf32, #gpu.address_space<global>>, vector<4xf32>
        %600 = arith.addi %13, %c2 : index
        %601 = vector.extract %27[2] : vector<4xf32> from vector<4x4xf32>
        vector.store %601, %2[%600, %18] : memref<4096x4096xf32, #gpu.address_space<global>>, vector<4xf32>
        %602 = arith.addi %13, %c3 : index
        %603 = vector.extract %27[3] : vector<4xf32> from vector<4x4xf32>
        vector.store %603, %2[%602, %18] : memref<4096x4096xf32, #gpu.address_space<global>>, vector<4xf32>
        return
      }
    }
  }
}

// CHECK-LABEL: llvm.func @matmul_dispatch_0
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "s_waitcnt lgkmcnt(0)\0As_barrier", ""  : () -> ()
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "s_waitcnt lgkmcnt(0)\0As_barrier", ""  : () -> ()
