# Isotropic Diffusion on 3D Grids

Two inner loops (depth 2 and 3) of optimized vertex
```C++
for (std::size_t y = 1; y < width + 1; ++y) {
    const float2 * __restrict__ top    = (float2 *) &in[(y+0) + (x-1)*in_width][0];
    const float2 * __restrict__ left   = (float2 *) &in[(y-1) + (x+0)*in_width][0];
    const float2 * __restrict__ middle = (float2 *) &in[(y+0) + (x+0)*in_width][0];
    const float2 * __restrict__ right  = (float2 *) &in[(y+1) + (x+0)*in_width][0];
    const float2 * __restrict__ bottom = (float2 *) &in[(y+0) + (x+1)*in_width][0];
    float * __restrict__ output = (float *) &out[(y-1) + (x-1)*width][0];

    for (std::size_t z = 1; z < half_depth; ++z) { // z=(1,2),(3,4),... in output
        
        //[z-1] [z+0] [z+1]
        //[x y] [x y] [x y] 

        temp.x = middle[z-1].y + middle[z].y; // front + back elem. for stencil 0
        temp.y = middle[z].x + middle[z+1].x; // front + back elem. for stencil 1
        
        temp = alpha*middle[z] + step_size*(temp + top[z] + bottom[z] + left[z] + right[z]);
        output[2*z - 1] = temp.x; // Output for left stencil 0
        output[2*z - 0] = temp.y; // Output for right stencil 1
    }
}
```

Corresponding assembly to the innermost loop (compiled with popc)
```
.LBB3_23:                               #   Parent Loop BB3_7 Depth=1
                                        #     Parent Loop BB3_21 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add $m3, $m15, -12
	ld32 $a4, $m4, $m3, 0
	add $m3, $m15, -8
	ld64 $a2:3, $m4, $m3, 0
	{
		ld32step $a5, $m15, $m4+=, 2
		f32add $a4, $a4, $a3
	}
	{
		ld64step $a6:7, $m15, $m7+=, 1
		f32add $a5, $a2, $a5
	}
	{
		ld64step $a6:7, $m15, $m9+=, 1
		f32v2add $a4:5, $a6:7, $a4:5
	}
	{
		ld64step $a6:7, $m15, $m5+=, 1
		f32v2add $a4:5, $a6:7, $a4:5
	}
	{
		ld64step $a6:7, $m15, $m10+=, 1
		f32v2add $a4:5, $a6:7, $a4:5
	}
	{
		ld32 $a6, $m0, $m15, 7
		f32v2add $a4:5, $a6:7, $a4:5
	}
	f32v2mul $a4:5, $a6:B, $a4:5
	f32v2mul $a2:3, $a0:1, $a2:3
	f32v2add $a2:3, $a2:3, $a4:5
	st32 $a3, $m6, $m15, 1
	st32step $a2, $m15, $m6+=, 2
	brnzdec $m1, .LBB3_23
	bri .LBB3_16
```