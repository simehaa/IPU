# Isotropic Diffusion on 2D Grids

Two inner loops (depth 1 and 2) of optimized vertex
```C++
for (int i = 1; i < height + 1; ++i) {
    const float2 * __restrict__ north  = (float2 *) &in[i - 1][0];
    const float2 * __restrict__ middle = (float2 *) &in[i + 0][0];
    const float2 * __restrict__ south  = (float2 *) &in[i + 1][0];
    float * __restrict__ output = (float *) &out[i - 1][0];
    // Stencil 0 (left) and stencil 1 (right)
    //      a   b
    //  c   d   e   f
    //      g   h
    for (int j = 1; j < half_width; ++j) {
        temp.x = middle[j - 1].y + middle[j].y; // c and e (sides for left stencil)
        temp.y = middle[j].x + middle[j + 1].x; // d and f (sides for right stencil)
        // Calculate output vectorized, by using a float2 computation
        temp = gamma*middle[j] + alpha*(north[j] + temp + south[j]);
        output[2*j - 1] = temp.x; // Output for left stencil
        output[2*j - 0] = temp.y; // Output for right stencil
    }
}
```

Corresponding assembly to the innermost loop (compiled with popc)
```
.LBB2_9:                                #   Parent Loop BB2_8 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	ld32 $a2, $m7, $m4, 4
	ld64 $a4:5, $m7, $m4, 1
	{
		ld32 $a2, $m7, $m4, 1
		f32add $a3, $a4, $a2
	}
	{
		ld64 $a6:7, $m10, $m4, 0
		f32add $a2, $a2, $a5
	}
	{
		ld64 $a6:7, $m6, $m4, 1
		f32v2add $a2:3, $a6:7, $a2:3
	}
	{
		ld32 $a6, $m0, $m15, 6
		f32v2add $a2:3, $a6:7, $a2:3
	}
	f32v2mul $a2:3, $a6:B, $a2:3
	f32v2mul $a4:5, $a0:1, $a4:5
	f32v2add $a2:3, $a4:5, $a2:3
	st32 $a3, $m9, $m4, 1
	st32 $a2, $m9, $m4, 0
	add $m4, $m4, 8
	brnzdec $m2, .LBB2_9
```