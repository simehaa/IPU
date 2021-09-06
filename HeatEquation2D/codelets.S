	.text
	.allow_optimizations
	.file	"3"
	.section	.text.__runCodelet_HeatEquationSimple,"ax",@progbits
	.globl	__runCodelet_HeatEquationSimple # -- Begin function __runCodelet_HeatEquationSimple
	.p2align	2
	.type	__runCodelet_HeatEquationSimple,@function
__runCodelet_HeatEquationSimple:        # @__runCodelet_HeatEquationSimple
.Lfunc_begin0:
# %bb.0:
	setzi $m6, 88
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	add $m11, $m12, -88
	ld32 $m0, $m13, $m15, 4
	cmpslt $m1, $m0, 1
	brnz $m1, .LBB0_16
# %bb.1:
	ld32 $m7, $m13, $m15, 5
	cmpslt $m1, $m7, 1
	brnz $m1, .LBB0_16
# %bb.2:
	{
		ld32 $a0, $m13, $m15, 6
		or $a1, $a15, 3229614080
	}
	{
		andc $m1, $m7, 1
		f32mul $a0, $a0, $a1
	}
	{
		ld32 $m2, $m13, $m15, 0
		or $a1, $a15, 1065353216
	}
	f32add $a6, $a0, $a1
	{
		st32 $m1, $m11, $m15, 3
		mov	$a7, $a6
	}
	add $m1, $m1, -2
	shr $m1, $m1, 1
	add $m1, $m1, 1
	st32 $m1, $m11, $m15, 2                 # 4-byte Folded Spill
	ld32 $m1, $m2, $m15, 0
	st32 $m1, $m11, $m15, 13                # 4-byte Folded Spill
	ld32 $m1, $m13, $m15, 2
	st32 $m1, $m11, $m15, 9                 # 4-byte Folded Spill
	st32 $m2, $m11, $m15, 10                # 4-byte Folded Spill
	ld32 $m1, $m2, $m15, 2
	st32 $m1, $m11, $m15, 14                # 4-byte Folded Spill
	shl $m1, $m7, 2
	setzi $m2, 1
	add $m8, $m13, 24
	add $m3, $m1, 4
	st32 $m3, $m11, $m15, 11                # 4-byte Folded Spill
	add $m3, $m13, 25
	st32 $m3, $m11, $m15, 6                 # 4-byte Folded Spill
	st32 $m1, $m11, $m15, 8                 # 4-byte Folded Spill
	add $m1, $m1, 8
	st32 $m1, $m11, $m15, 5                 # 4-byte Folded Spill
	or $m1, $m7, 1
	st32 $m1, $m11, $m15, 1                 # 4-byte Folded Spill
	add $m1, $m7, 1
	st32 $m1, $m11, $m15, 7                 # 4-byte Folded Spill
	st32 $m2, $m11, $m15, 15                # 4-byte Folded Spill
	st32 $m7, $m11, $m15, 4                 # 4-byte Folded Spill
	st64 $a6:7, $m11, $m15, 9               # 8-byte Folded Spill
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_11 Depth 2
                                        #     Child Loop BB0_14 Depth 2
	shl $m1, $m2, 3
	add $m2, $m2, 1
	cmpult $m3, $m7, 2
	st32 $m2, $m11, $m15, 12                # 4-byte Folded Spill
	shl $m2, $m2, 3
	ld32 $m4, $m11, $m15, 9                 # 4-byte Folded Reload
	add $m1, $m1, $m4
	add $m4, $m15, -8
	ld32 $m9, $m11, $m15, 14                # 4-byte Folded Reload
	ld32 $m5, $m11, $m15, 10                # 4-byte Folded Reload
	ld32 $m2, $m5, $m2, 0
	st32 $m2, $m11, $m15, 14                # 4-byte Folded Spill
	ld32 $m2, $m1, $m4, 0
	ld32 $m6, $m11, $m15, 15                # 4-byte Folded Reload
	brz $m3, .LBB0_4
.LBB0_13:                               #   in Loop: Header=BB0_3 Depth=1
	shl $m5, $m6, 2
	ld32 $m1, $m11, $m15, 7                 # 4-byte Folded Reload
	sub $m1, $m1, $m6
	add $m1, $m1, -1
	add $m3, $m2, $m5
	ld32 $m2, $m11, $m15, 13                # 4-byte Folded Reload
	add $m2, $m2, $m5
	add $m3, $m3, -4
	add $m4, $m9, $m5
	ld32 $m6, $m11, $m15, 14                # 4-byte Folded Reload
	add $m5, $m6, $m5
.LBB0_14:                               #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add $m6, $m15, -4
	ld32 $a0, $m4, $m6, 0
	ld32step $a1, $m15, $m4+=, 1
	{
		ld32 $a2, $m8, $m15, 0
		f32mul $a1, $a6, $a1
	}
	ld32step $a3, $m15, $m2+=, 1
	ld32step $a4, $m15, $m5+=, 1
	{
		ld32 $a4, $m4, $m15, 0
		f32add $a3, $a3, $a4
	}
	f32add $a0, $a3, $a0
	f32add $a0, $a0, $a4
	f32mul $a0, $a2, $a0
	f32add $a0, $a1, $a0
	st32step $a0, $m15, $m3+=, 1
	brnzdec $m1, .LBB0_14
.LBB0_15:                               #   in Loop: Header=BB0_3 Depth=1
	add $m0, $m0, -1
	st32 $m9, $m11, $m15, 13                # 4-byte Folded Spill
	ld32 $m2, $m11, $m15, 12                # 4-byte Folded Reload
	brnz $m0, .LBB0_3
	bri .LBB0_16
.LBB0_4:                                #   in Loop: Header=BB0_3 Depth=1
	ld32 $m1, $m11, $m15, 8                 # 4-byte Folded Reload
	add $m4, $m2, $m1
	add $m1, $m9, 4
	ld32 $m3, $m11, $m15, 11                # 4-byte Folded Reload
	add $m3, $m9, $m3
	cmpult $m3, $m2, $m3
	cmpult $m1, $m1, $m4
	and $m1, $m3, $m1
	ld32 $m6, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m1, .LBB0_13
# %bb.5:                                #   in Loop: Header=BB0_3 Depth=1
	ld32 $m1, $m11, $m15, 6                 # 4-byte Folded Reload
	cmpult $m1, $m2, $m1
	cmpult $m3, $m8, $m4
	and $m1, $m1, $m3
	ld32 $m6, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m1, .LBB0_13
# %bb.6:                                #   in Loop: Header=BB0_3 Depth=1
	ld32 $m5, $m11, $m15, 13                # 4-byte Folded Reload
	add $m3, $m5, 4
	ld32 $m1, $m11, $m15, 11                # 4-byte Folded Reload
	add $m1, $m5, $m1
	cmpult $m1, $m2, $m1
	cmpult $m6, $m3, $m4
	and $m1, $m1, $m6
	ld32 $m6, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m1, .LBB0_13
# %bb.7:                                #   in Loop: Header=BB0_3 Depth=1
	ld32 $m5, $m11, $m15, 14                # 4-byte Folded Reload
	add $m1, $m5, 4
	ld32 $m6, $m11, $m15, 11                # 4-byte Folded Reload
	add $m6, $m5, $m6
	cmpult $m6, $m2, $m6
	cmpult $m10, $m1, $m4
	and $m10, $m6, $m10
	ld32 $m6, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m10, .LBB0_13
# %bb.8:                                #   in Loop: Header=BB0_3 Depth=1
	ld32 $m5, $m11, $m15, 8                 # 4-byte Folded Reload
	add $m6, $m9, $m5
	cmpult $m6, $m2, $m6
	cmpult $m10, $m9, $m4
	and $m10, $m6, $m10
	ld32 $m6, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m10, .LBB0_13
# %bb.9:                                #   in Loop: Header=BB0_3 Depth=1
	add $m10, $m9, 8
	ld32 $m5, $m11, $m15, 5                 # 4-byte Folded Reload
	add $m6, $m9, $m5
	cmpult $m6, $m2, $m6
	cmpult $m4, $m10, $m4
	and $m4, $m6, $m4
	ld32 $m6, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m4, .LBB0_13
# %bb.10:                               #   in Loop: Header=BB0_3 Depth=1
	ld32 $m4, $m11, $m15, 2                 # 4-byte Folded Reload
	ld32 $a0, $m8, $m15, 0
	{
		add $m4, $m4, -1
		mov	$a1, $a0
	}
	st64 $a0:1, $m11, $m15, 8               # 8-byte Folded Spill
	mov	$m6, $m2
.LBB0_11:                               #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add $m7, $m15, -4
	add $m5, $m15, -8
	ld32 $a2, $m10, $m7, 0
	ld32 $a6, $m10, $m5, 0
	ld32 $a5, $m10, $m15, 1
	ld32step $a3, $m15, $m10+=, 2
	ld64 $a0:1, $m11, $m15, 9               # 8-byte Folded Reload
	f32v2mul $a0:1, $a0:1, $a2:3
	{
		st64 $a0:1, $m11, $m15, 10
		mov	$a7, $a2
	}
	{
		ld32 $a3, $m3, $m15, 1
		mov	$a4, $a3
	}
	ld32step $a2, $m15, $m3+=, 2
	ld32 $a1, $m1, $m15, 1
	ld32step $a0, $m15, $m1+=, 2
	f32v2add $a0:1, $a2:3, $a0:1
	f32v2add $a0:1, $a0:1, $a6:7
	{
		ld64 $a2:3, $m11, $m15, 8
		f32v2add $a0:1, $a0:1, $a4:5
	}
	{
		ld64 $a2:3, $m11, $m15, 10
		f32v2mul $a0:1, $a2:3, $a0:1
	}
	f32v2add $a0:1, $a2:3, $a0:1
	st32 $a1, $m6, $m15, 1
	st32step $a0, $m15, $m6+=, 2
	brnzdec $m4, .LBB0_11
# %bb.12:                               #   in Loop: Header=BB0_3 Depth=1
	ld32 $m7, $m11, $m15, 4                 # 4-byte Folded Reload
	ld32 $m1, $m11, $m15, 3                 # 4-byte Folded Reload
	cmpeq $m1, $m7, $m1
	ld32 $m6, $m11, $m15, 1                 # 4-byte Folded Reload
	ld64 $a6:7, $m11, $m15, 9               # 8-byte Folded Reload
	brz $m1, .LBB0_13
	bri .LBB0_15
.LBB0_16:
	exitz $m15
.Lfunc_end0:
	.size	__runCodelet_HeatEquationSimple, .Lfunc_end0-__runCodelet_HeatEquationSimple
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_HeatEquationSimple
	.long	.Lfunc_begin0
	.byte	88
	.section	.text.__runCodelet_HeatEquationSimple,"ax",@progbits
                                        # -- End function
	.section	.text.__runCodelet_HeatEquationOptimized,"ax",@progbits
	.globl	__runCodelet_HeatEquationOptimized # -- Begin function __runCodelet_HeatEquationOptimized
	.p2align	2
	.type	__runCodelet_HeatEquationOptimized,@function
__runCodelet_HeatEquationOptimized:     # @__runCodelet_HeatEquationOptimized
.Lfunc_begin1:
# %bb.0:
	mov	$m11, $m12
	mov	$m0, $m13
	call $m10, _ZN21HeatEquationOptimized7computeEv
	exitnz $m0
.Lfunc_end1:
	.size	__runCodelet_HeatEquationOptimized, .Lfunc_end1-__runCodelet_HeatEquationOptimized
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_HeatEquationOptimized
	.long	.Lfunc_begin1
	.byte	0
	.section	.text.__runCodelet_HeatEquationOptimized,"ax",@progbits
                                        # -- End function
	.section	.text._ZN21HeatEquationOptimized7computeEv,"axG",@progbits,_ZN21HeatEquationOptimized7computeEv,comdat
	.weak	_ZN21HeatEquationOptimized7computeEv # -- Begin function _ZN21HeatEquationOptimized7computeEv
	.p2align	2
	.type	_ZN21HeatEquationOptimized7computeEv,@function
_ZN21HeatEquationOptimized7computeEv:   # @_ZN21HeatEquationOptimized7computeEv
.Lfunc_begin2:
# %bb.0:
	add $m11, $m11, -56
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st32 $m8, $m11, $m15, 7                 # 4-byte Folded Spill
	st32 $m9, $m11, $m15, 6                 # 4-byte Folded Spill
	st32 $m10, $m11, $m15, 5                # 4-byte Folded Spill
	st64 $a6:7, $m11, $m15, 1               # 8-byte Folded Spill
	st32 $m7, $m11, $m15, 1                 # 4-byte Folded Spill
	ld32 $m1, $m0, $m15, 4
	cmpslt $m5, $m1, 1
	brnz $m5, .LBB2_14
# %bb.1:
	{
		ld32 $m2, $m0, $m15, 5
		or $a0, $a15, 3229614080
	}
	ld32 $a1, $m0, $m15, 6
	{
		ld32 $m9, $m0, $m15, 0
		f32mul $a0, $a1, $a0
	}
	{
		and $m6, $m2, 1
		or $a2, $a15, 1065353216
	}
	{
		ld32 $m4, $m0, $m15, 2
		f32add $a6, $a0, $a2
	}
	ld32 $m3, $m9, $m15, 2
	st32 $m5, $m11, $m15, 10                # 4-byte Folded Spill
	brnz $m6, .LBB2_2
# %bb.5:
	add $m5, $m2, 1
	st32 $m5, $m11, $m15, 13                # 4-byte Folded Spill
	add $m7, $m2, -1
	add $m8, $m1, -1
	add $m9, $m9, 16
.LBB2_6:                                # =>This Inner Loop Header: Depth=1
	add $m10, $m15, -16
	mov	$m5, $m3
	ld32 $m3, $m9, $m15, 0
	ld32 $m10, $m9, $m10, 0
	ld32 $a2, $m3, $m15, 1
	ld32 $a3, $m10, $m15, 1
	{
		ld32 $a3, $m5, $m15, 2
		f32add $a2, $a2, $a3
	}
	{
		ld32 $a3, $m5, $m15, 0
		f32add $a2, $a2, $a3
	}
	{
		ld32 $a4, $m5, $m15, 1
		f32add $a2, $a2, $a3
	}
	{
		ld32 $m6, $m4, $m15, 0
		f32mul $a3, $a6, $a4
	}
	f32mul $a1, $a1, $a2
	f32add $a1, $a3, $a1
	st32 $a1, $m6, $m15, 0
	ld32 $a1, $m3, $m15, $m2
	ld32 $a2, $m10, $m15, $m2
	{
		ld32 $m10, $m11, $m15, 13
		f32add $a1, $a1, $a2
	}
	ld32 $a2, $m5, $m15, $m10
	{
		ld32 $a2, $m5, $m15, $m7
		f32add $a1, $a1, $a2
	}
	{
		ld32 $a3, $m5, $m15, $m2
		f32add $a1, $a1, $a2
	}
	{
		ld32 $a2, $m0, $m15, 6
		f32mul $a3, $a6, $a3
	}
	f32mul $a1, $a2, $a1
	f32add $a1, $a3, $a1
	st32 $a1, $m6, $m15, $m7
	brz $m8, .LBB2_8
# %bb.7:                                #   in Loop: Header=BB2_6 Depth=1
	ld32 $a1, $m0, $m15, 6
	add $m8, $m8, -1
	add $m4, $m4, 8
	add $m9, $m9, 8
	bri .LBB2_6
.LBB2_2:
	add $m6, $m1, -1
	add $m7, $m9, 16
.LBB2_3:                                # =>This Inner Loop Header: Depth=1
	add $m5, $m15, -16
	ld32 $a0, $m3, $m15, 1
	{
		ld32 $m5, $m7, $m5, 0
		f32mul $a0, $a6, $a0
	}
	ld32 $a2, $m3, $m15, 2
	ld32 $a3, $m3, $m15, 0
	ld32 $m3, $m7, $m15, 0
	ld32 $m8, $m4, $m15, 0
	ld32 $a4, $m3, $m15, 1
	ld32 $a5, $m5, $m15, 1
	f32add $a4, $a4, $a5
	f32add $a2, $a4, $a2
	f32add $a2, $a2, $a3
	f32mul $a1, $a1, $a2
	f32add $a0, $a0, $a1
	st32 $a0, $m8, $m15, 0
	brz $m6, .LBB2_8
# %bb.4:                                #   in Loop: Header=BB2_3 Depth=1
	ld32 $a1, $m0, $m15, 6
	add $m6, $m6, -1
	add $m7, $m7, 8
	add $m4, $m4, 8
	bri .LBB2_3
.LBB2_8:
	ld32 $m3, $m11, $m15, 10                # 4-byte Folded Reload
	brnz $m3, .LBB2_14
# %bb.9:
	shr $m3, $m2, 31
	add $m3, $m2, $m3
	andc $m5, $m3, 1
	shrs $m4, $m3, 1
	sub $m3, $m2, $m5
	add $m3, $m4, $m3
	cmpslt $m3, $m3, 2
	brnz $m3, .LBB2_14
# %bb.10:
	ld32 $m3, $m0, $m15, 0
	ld32 $m5, $m3, $m15, 0
	st32 $m3, $m11, $m15, 9                 # 4-byte Folded Spill
	ld32 $m6, $m3, $m15, 2
	ld32 $m3, $m0, $m15, 2
	st32 $m3, $m11, $m15, 8                 # 4-byte Folded Spill
	{
		xnor $m4, $m4, $m15
		mov	$a7, $a6
	}
	add $m2, $m4, $m2
	setzi $m7, 1
	st64 $a6:7, $m11, $m15, 5               # 8-byte Folded Spill
.LBB2_11:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_12 Depth 2
	mov	$m8, $m15
	shl $m4, $m7, 3
	add $m7, $m7, 1
	shl $m3, $m7, 3
	ld32 $m9, $m11, $m15, 8                 # 4-byte Folded Reload
	add $m4, $m4, $m9
	add $m9, $m15, -8
	ld32 $m4, $m4, $m9, 0
	mov	$m9, $m2
	add $m10, $m5, 8
	mov	$m5, $m6
	ld32 $m6, $m11, $m15, 9                 # 4-byte Folded Reload
	ld32 $m6, $m6, $m3, 0
	add $m9, $m9, -1
	add $m4, $m4, 4
.LBB2_12:                               #   Parent Loop BB2_11 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	ld32 $a2, $m5, $m8, 1
	ld32 $a3, $m5, $m8, 4
	ld32 $a0, $m0, $m15, 6
	st32 $a0, $m11, $m15, 13                # 4-byte Folded Spill
	ld64 $a6:7, $m5, $m8, 1
	{
		ld64 $a0:1, $m10, $m8, 0
		f32add $a2, $a2, $a7
	}
	{
		ld64 $a4:5, $m6, $m8, 1
		f32add $a3, $a6, $a3
	}
	f32v2add $a0:1, $a0:1, $a2:3
	{
		ld64 $a2:3, $m11, $m15, 5
		f32v2add $a0:1, $a4:5, $a0:1
	}
	{
		ld32 $a4, $m11, $m15, 13
		f32v2mul $a2:3, $a2:3, $a6:7
	}
	f32v2mul $a0:1, $a4:B, $a0:1
	f32v2add $a0:1, $a2:3, $a0:1
	st32 $a1, $m4, $m8, 1
	st32 $a0, $m4, $m8, 0
	add $m8, $m8, 8
	brnzdec $m9, .LBB2_12
# %bb.13:                               #   in Loop: Header=BB2_11 Depth=1
	add $m1, $m1, -1
	brnz $m1, .LBB2_11
.LBB2_14:
	setzi $m0, 1
	ld32 $m7, $m11, $m15, 1                 # 4-byte Folded Reload
	ld64 $a6:7, $m11, $m15, 1               # 8-byte Folded Reload
	ld32 $m10, $m11, $m15, 5                # 4-byte Folded Reload
	ld32 $m9, $m11, $m15, 6                 # 4-byte Folded Reload
	ld32 $m8, $m11, $m15, 7                 # 4-byte Folded Reload
	add $m11, $m11, 56
	br $m10
.Lfunc_end2:
	.size	_ZN21HeatEquationOptimized7computeEv, .Lfunc_end2-_ZN21HeatEquationOptimized7computeEv
	.section	.stack_sizes,"Go",@progbits,_ZN21HeatEquationOptimized7computeEv,comdat,.text._ZN21HeatEquationOptimized7computeEv
	.long	.Lfunc_begin2
	.byte	56
	.section	.text._ZN21HeatEquationOptimized7computeEv,"axG",@progbits,_ZN21HeatEquationOptimized7computeEv,comdat
                                        # -- End function
	.ident	"clang version 13.0.0 (ssh://git@phabricator.sourcevertex.net/diffusion/LLVMPROJECT/llvm-project.git 6eeafc711ed2cca929acfbe560c20764774f9a28)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
