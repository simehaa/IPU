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
	mov	$m11, $m12
	mov	$m0, $m13
	call $m10, _ZN18HeatEquationSimple7computeEv
	exitnz $m0
.Lfunc_end0:
	.size	__runCodelet_HeatEquationSimple, .Lfunc_end0-__runCodelet_HeatEquationSimple
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_HeatEquationSimple
	.long	.Lfunc_begin0
	.byte	0
	.section	.text.__runCodelet_HeatEquationSimple,"ax",@progbits
                                        # -- End function
	.section	.text._ZN18HeatEquationSimple7computeEv,"axG",@progbits,_ZN18HeatEquationSimple7computeEv,comdat
	.weak	_ZN18HeatEquationSimple7computeEv # -- Begin function _ZN18HeatEquationSimple7computeEv
	.p2align	2
	.type	_ZN18HeatEquationSimple7computeEv,@function
_ZN18HeatEquationSimple7computeEv:      # @_ZN18HeatEquationSimple7computeEv
.Lfunc_begin1:
# %bb.0:
	add $m11, $m11, -160
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st32 $m8, $m11, $m15, 7                 # 4-byte Folded Spill
	st32 $m9, $m11, $m15, 6                 # 4-byte Folded Spill
	st32 $m10, $m11, $m15, 5                # 4-byte Folded Spill
	st64 $a6:7, $m11, $m15, 1               # 8-byte Folded Spill
	st32 $m7, $m11, $m15, 1                 # 4-byte Folded Spill
	ld32 $m1, $m0, $m15, 4
	add $m4, $m1, 1
	cmpult $m1, $m4, 2
	brnz $m1, .LBB1_21
# %bb.1:
	{
		ld32 $a0, $m0, $m15, 7
		or $a1, $a15, 3233808384
	}
	{
		ld32 $m5, $m0, $m15, 5
		f32mul $a0, $a0, $a1
	}
	{
		setzi $m2, 1
		or $a1, $a15, 1065353216
	}
	{
		add $m1, $m5, 1
		f32add $a6, $a0, $a1
	}
	add $m3, $m0, 28
	st32 $m5, $m11, $m15, 10                # 4-byte Folded Spill
	add $m5, $m5, 2
	st32 $m5, $m11, $m15, 9                 # 4-byte Folded Spill
	add $m5, $m0, 29
	{
		st32 $m5, $m11, $m15, 17
		mov	$a7, $a6
	}
	cmpult $m5, $m2, $m1
	st32 $m0, $m11, $m15, 12                # 4-byte Folded Spill
	st32 $m4, $m11, $m15, 11                # 4-byte Folded Spill
	st32 $m5, $m11, $m15, 8                 # 4-byte Folded Spill
	bri .LBB1_2
.LBB1_3:                                #   in Loop: Header=BB1_2 Depth=1
	add $m6, $m2, 1
.LBB1_20:                               #   in Loop: Header=BB1_2 Depth=1
	cmpeq $m1, $m6, $m4
	mov	$m2, $m6
	brnz $m1, .LBB1_21
.LBB1_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_6 Depth 2
                                        #       Child Loop BB1_14 Depth 3
                                        #       Child Loop BB1_17 Depth 3
	brz $m5, .LBB1_3
# %bb.4:                                #   in Loop: Header=BB1_2 Depth=1
	ld32 $m7, $m0, $m15, 6
	add $m8, $m7, 1
	cmpult $m1, $m8, 2
	add $m6, $m2, 1
	brnz $m1, .LBB1_20
# %bb.5:                                #   in Loop: Header=BB1_2 Depth=1
	st32 $m8, $m11, $m15, 20                # 4-byte Folded Spill
	andc $m1, $m7, 1
	st32 $m1, $m11, $m15, 16                # 4-byte Folded Spill
	add $m1, $m1, -2
	shr $m1, $m1, 1
	add $m1, $m1, 1
	st32 $m1, $m11, $m15, 15                # 4-byte Folded Spill
	ld32 $m4, $m11, $m15, 10                # 4-byte Folded Reload
	mov	$m8, $m4
	add $m1, $m2, -1
	ld32 $m5, $m11, $m15, 9                 # 4-byte Folded Reload
	mul $m2, $m2, $m5
	st32 $m2, $m11, $m15, 26                # 4-byte Folded Spill
	ld32 $m2, $m0, $m15, 0
	st32 $m2, $m11, $m15, 25                # 4-byte Folded Spill
	ld32 $m0, $m0, $m15, 2
	st32 $m0, $m11, $m15, 24                # 4-byte Folded Spill
	setzi $m0, 1
	st32 $m6, $m11, $m15, 13                # 4-byte Folded Spill
	mul $m2, $m6, $m5
	st32 $m2, $m11, $m15, 23                # 4-byte Folded Spill
	mul $m2, $m1, $m5
	st32 $m2, $m11, $m15, 22                # 4-byte Folded Spill
	mul $m1, $m1, $m4
	st32 $m1, $m11, $m15, 21                # 4-byte Folded Spill
	shl $m1, $m7, 2
	st32 $m1, $m11, $m15, 31                # 4-byte Folded Spill
	or $m1, $m7, 1
	st32 $m1, $m11, $m15, 14                # 4-byte Folded Spill
	st32 $m0, $m11, $m15, 35                # 4-byte Folded Spill
	mov	$m5, $m0
	st32 $m7, $m11, $m15, 27                # 4-byte Folded Spill
.LBB1_6:                                #   Parent Loop BB1_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_14 Depth 3
                                        #       Child Loop BB1_17 Depth 3
	st32 $m8, $m11, $m15, 29                # 4-byte Folded Spill
	add $m1, $m5, -1
	ld32 $m0, $m11, $m15, 26                # 4-byte Folded Reload
	add $m2, $m5, $m0
	shl $m2, $m2, 3
	ld32 $m4, $m11, $m15, 23                # 4-byte Folded Reload
	add $m4, $m5, $m4
	shl $m4, $m4, 3
	ld32 $m6, $m11, $m15, 25                # 4-byte Folded Reload
	ld32 $m4, $m6, $m4, 0
	st32 $m4, $m11, $m15, 34                # 4-byte Folded Spill
	ld32 $m4, $m11, $m15, 22                # 4-byte Folded Reload
	st32 $m5, $m11, $m15, 28                # 4-byte Folded Spill
	add $m4, $m5, $m4
	shl $m4, $m4, 3
	ld32 $m4, $m6, $m4, 0
	st32 $m4, $m11, $m15, 33                # 4-byte Folded Spill
	add $m4, $m1, $m0
	ld32 $m0, $m11, $m15, 21                # 4-byte Folded Reload
	add $m1, $m1, $m0
	shl $m4, $m4, 3
	shl $m1, $m1, 3
	ld32 $m0, $m6, $m4, 0
	st32 $m0, $m11, $m15, 30                # 4-byte Folded Spill
	ld32 $m0, $m11, $m15, 24                # 4-byte Folded Reload
	ld32 $m8, $m0, $m1, 0
	cmpult $m5, $m7, 2
	ld32 $m1, $m6, $m2, 0
	ld32 $m0, $m6, $m2, 2
	st32 $m0, $m11, $m15, 32                # 4-byte Folded Spill
	ld32 $m4, $m11, $m15, 35                # 4-byte Folded Reload
	brz $m5, .LBB1_7
.LBB1_16:                               #   in Loop: Header=BB1_6 Depth=2
	shl $m0, $m4, 2
	mov	$m5, $m1
	ld32 $m1, $m11, $m15, 20                # 4-byte Folded Reload
	sub $m2, $m1, $m4
	add $m2, $m2, -1
	add $m1, $m8, $m0
	add $m1, $m1, -4
	ld32 $m4, $m11, $m15, 34                # 4-byte Folded Reload
	add $m4, $m4, $m0
	add $m5, $m5, $m0
	ld32 $m6, $m11, $m15, 30                # 4-byte Folded Reload
	add $m6, $m6, $m0
	ld32 $m7, $m11, $m15, 32                # 4-byte Folded Reload
	add $m7, $m7, $m0
	ld32 $m8, $m11, $m15, 33                # 4-byte Folded Reload
	add $m8, $m8, $m0
.LBB1_17:                               #   Parent Loop BB1_2 Depth=1
                                        #     Parent Loop BB1_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add $m0, $m15, -4
	ld32 $a0, $m5, $m0, 0
	ld32step $a1, $m15, $m5+=, 1
	{
		ld32 $a2, $m3, $m15, 0
		f32mul $a1, $a6, $a1
	}
	ld32step $a3, $m15, $m4+=, 1
	ld32step $a4, $m15, $m8+=, 1
	{
		ld32step $a4, $m15, $m7+=, 1
		f32add $a3, $a3, $a4
	}
	{
		ld32step $a4, $m15, $m6+=, 1
		f32add $a3, $a3, $a4
	}
	{
		ld32 $a4, $m5, $m15, 0
		f32add $a3, $a3, $a4
	}
	f32add $a3, $a3, $a4
	f32add $a0, $a3, $a0
	f32mul $a0, $a2, $a0
	f32add $a0, $a1, $a0
	st32step $a0, $m15, $m1+=, 1
	brnzdec $m2, .LBB1_17
.LBB1_18:                               #   in Loop: Header=BB1_6 Depth=2
	ld32 $m8, $m11, $m15, 29                # 4-byte Folded Reload
	add $m8, $m8, -1
	ld32 $m5, $m11, $m15, 28                # 4-byte Folded Reload
	add $m5, $m5, 1
	ld32 $m7, $m11, $m15, 27                # 4-byte Folded Reload
	brnz $m8, .LBB1_6
	bri .LBB1_19
.LBB1_7:                                #   in Loop: Header=BB1_6 Depth=2
	ld32 $m0, $m11, $m15, 31                # 4-byte Folded Reload
	add $m4, $m1, $m0
	add $m2, $m8, $m0
	add $m4, $m4, 8
	cmpult $m4, $m8, $m4
	cmpult $m5, $m1, $m2
	and $m5, $m4, $m5
	ld32 $m4, $m11, $m15, 35                # 4-byte Folded Reload
	brnz $m5, .LBB1_16
# %bb.8:                                #   in Loop: Header=BB1_6 Depth=2
	ld32 $m0, $m11, $m15, 17                # 4-byte Folded Reload
	cmpult $m4, $m8, $m0
	cmpult $m5, $m3, $m2
	and $m5, $m4, $m5
	ld32 $m4, $m11, $m15, 35                # 4-byte Folded Reload
	brnz $m5, .LBB1_16
# %bb.9:                                #   in Loop: Header=BB1_6 Depth=2
	ld32 $m0, $m11, $m15, 34                # 4-byte Folded Reload
	add $m5, $m0, 4
	ld32 $m0, $m11, $m15, 31                # 4-byte Folded Reload
	add $m4, $m5, $m0
	cmpult $m4, $m8, $m4
	cmpult $m6, $m5, $m2
	and $m6, $m4, $m6
	ld32 $m4, $m11, $m15, 35                # 4-byte Folded Reload
	brnz $m6, .LBB1_16
# %bb.10:                               #   in Loop: Header=BB1_6 Depth=2
	ld32 $m0, $m11, $m15, 33                # 4-byte Folded Reload
	add $m10, $m0, 4
	ld32 $m0, $m11, $m15, 31                # 4-byte Folded Reload
	add $m4, $m10, $m0
	cmpult $m4, $m8, $m4
	cmpult $m6, $m10, $m2
	and $m6, $m4, $m6
	ld32 $m4, $m11, $m15, 35                # 4-byte Folded Reload
	brnz $m6, .LBB1_16
# %bb.11:                               #   in Loop: Header=BB1_6 Depth=2
	ld32 $m0, $m11, $m15, 32                # 4-byte Folded Reload
	add $m7, $m0, 4
	ld32 $m0, $m11, $m15, 31                # 4-byte Folded Reload
	add $m4, $m7, $m0
	cmpult $m4, $m8, $m4
	cmpult $m6, $m7, $m2
	and $m6, $m4, $m6
	ld32 $m4, $m11, $m15, 35                # 4-byte Folded Reload
	brnz $m6, .LBB1_16
# %bb.12:                               #   in Loop: Header=BB1_6 Depth=2
	ld32 $m0, $m11, $m15, 30                # 4-byte Folded Reload
	add $m9, $m0, 4
	ld32 $m0, $m11, $m15, 31                # 4-byte Folded Reload
	add $m4, $m9, $m0
	cmpult $m4, $m8, $m4
	cmpult $m2, $m9, $m2
	and $m2, $m4, $m2
	ld32 $m4, $m11, $m15, 35                # 4-byte Folded Reload
	brnz $m2, .LBB1_16
# %bb.13:                               #   in Loop: Header=BB1_6 Depth=2
	ld32 $m2, $m11, $m15, 15                # 4-byte Folded Reload
	st32 $m1, $m11, $m15, 19                # 4-byte Folded Spill
	add $m4, $m1, 8
	add $m2, $m2, -1
	st32 $m8, $m11, $m15, 18                # 4-byte Folded Spill
	mov	$m6, $m8
	mov	$m8, $m4
.LBB1_14:                               #   Parent Loop BB1_2 Depth=1
                                        #     Parent Loop BB1_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add $m1, $m15, -4
	add $m0, $m15, -8
	ld32 $a4, $m4, $m1, 0
	ld64 $a0:1, $m4, $m0, 0
	st64 $a0:1, $m11, $m15, 19              # 8-byte Folded Spill
	ld32step $a5, $m15, $m8+=, 2
	f32v2mul $a0:1, $a6:7, $a4:5
	{
		st64 $a0:1, $m11, $m15, 18
		mov	$a4:5, $a6:7
	}
	ld32 $a6, $m3, $m15, 0
	ld32 $a1, $m5, $m15, 1
	ld32step $a0, $m15, $m5+=, 2
	ld32 $a3, $m10, $m15, 1
	ld32step $a2, $m15, $m10+=, 2
	{
		ld32 $a3, $m7, $m15, 1
		f32v2add $a0:1, $a0:1, $a2:3
	}
	ld32step $a2, $m15, $m7+=, 2
	{
		ld32 $a3, $m9, $m15, 1
		f32v2add $a0:1, $a0:1, $a2:3
	}
	ld32step $a2, $m15, $m9+=, 2
	{
		ld64 $a2:3, $m4, $m15, 0
		f32v2add $a0:1, $a0:1, $a2:3
	}
	{
		mov	$m4, $m8
		f32v2add $a0:1, $a0:1, $a2:3
	}
	ld64 $a2:3, $m11, $m15, 19              # 8-byte Folded Reload
	f32v2add $a0:1, $a0:1, $a2:3
	f32v2mul $a0:1, $a6:B, $a0:1
	{
		ld64 $a2:3, $m11, $m15, 18
		mov	$a6:7, $a4:5
	}
	f32v2add $a0:1, $a2:3, $a0:1
	st32 $a1, $m6, $m15, 1
	st32step $a0, $m15, $m6+=, 2
	brnzdec $m2, .LBB1_14
# %bb.15:                               #   in Loop: Header=BB1_6 Depth=2
	ld32 $m0, $m11, $m15, 27                # 4-byte Folded Reload
	ld32 $m1, $m11, $m15, 16                # 4-byte Folded Reload
	cmpeq $m0, $m0, $m1
	ld32 $m4, $m11, $m15, 14                # 4-byte Folded Reload
	ld32 $m1, $m11, $m15, 19                # 4-byte Folded Reload
	ld32 $m8, $m11, $m15, 18                # 4-byte Folded Reload
	brz $m0, .LBB1_16
	bri .LBB1_18
.LBB1_19:                               #   in Loop: Header=BB1_2 Depth=1
	ld32 $m0, $m11, $m15, 12                # 4-byte Folded Reload
	ld32 $m4, $m11, $m15, 11                # 4-byte Folded Reload
	ld32 $m5, $m11, $m15, 8                 # 4-byte Folded Reload
	ld32 $m6, $m11, $m15, 13                # 4-byte Folded Reload
	bri .LBB1_20
.LBB1_21:
	setzi $m0, 1
	ld32 $m7, $m11, $m15, 1                 # 4-byte Folded Reload
	ld64 $a6:7, $m11, $m15, 1               # 8-byte Folded Reload
	ld32 $m10, $m11, $m15, 5                # 4-byte Folded Reload
	ld32 $m9, $m11, $m15, 6                 # 4-byte Folded Reload
	ld32 $m8, $m11, $m15, 7                 # 4-byte Folded Reload
	add $m11, $m11, 160
	br $m10
.Lfunc_end1:
	.size	_ZN18HeatEquationSimple7computeEv, .Lfunc_end1-_ZN18HeatEquationSimple7computeEv
	.section	.stack_sizes,"Go",@progbits,_ZN18HeatEquationSimple7computeEv,comdat,.text._ZN18HeatEquationSimple7computeEv
	.long	.Lfunc_begin1
	.ascii	"\240\001"
	.section	.text._ZN18HeatEquationSimple7computeEv,"axG",@progbits,_ZN18HeatEquationSimple7computeEv,comdat
                                        # -- End function
	.section	.text.__runCodelet_HeatEquationOptimized,"ax",@progbits
	.globl	__runCodelet_HeatEquationOptimized # -- Begin function __runCodelet_HeatEquationOptimized
	.p2align	2
	.type	__runCodelet_HeatEquationOptimized,@function
__runCodelet_HeatEquationOptimized:     # @__runCodelet_HeatEquationOptimized
.Lfunc_begin2:
# %bb.0:
	mov	$m11, $m12
	mov	$m0, $m13
	call $m10, _ZN21HeatEquationOptimized7computeEv
	exitnz $m0
.Lfunc_end2:
	.size	__runCodelet_HeatEquationOptimized, .Lfunc_end2-__runCodelet_HeatEquationOptimized
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_HeatEquationOptimized
	.long	.Lfunc_begin2
	.byte	0
	.section	.text.__runCodelet_HeatEquationOptimized,"ax",@progbits
                                        # -- End function
	.section	.text._ZN21HeatEquationOptimized7computeEv,"axG",@progbits,_ZN21HeatEquationOptimized7computeEv,comdat
	.weak	_ZN21HeatEquationOptimized7computeEv # -- Begin function _ZN21HeatEquationOptimized7computeEv
	.p2align	2
	.type	_ZN21HeatEquationOptimized7computeEv,@function
_ZN21HeatEquationOptimized7computeEv:   # @_ZN21HeatEquationOptimized7computeEv
.Lfunc_begin3:
# %bb.0:
	add $m11, $m11, -96
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st32 $m8, $m11, $m15, 5                 # 4-byte Folded Spill
	st32 $m9, $m11, $m15, 4                 # 4-byte Folded Spill
	st32 $m10, $m11, $m15, 3                # 4-byte Folded Spill
	st32 $a6, $m11, $m15, 2                 # 4-byte Folded Spill
	st32 $m7, $m11, $m15, 1                 # 4-byte Folded Spill
	ld32 $m1, $m0, $m15, 4
	add $m3, $m1, 1
	cmpult $m1, $m3, 2
	brnz $m1, .LBB3_23
# %bb.1:
	st32 $m1, $m11, $m15, 7                 # 4-byte Folded Spill
	{
		ld32 $a0, $m0, $m15, 7
		or $a1, $a15, 3233808384
	}
	{
		ld32 $m2, $m0, $m15, 5
		f32mul $a0, $a0, $a1
	}
	{
		ld32 $m6, $m0, $m15, 6
		or $a1, $a15, 1065353216
	}
	{
		st32 $m6, $m11, $m15, 22
		f32add $a0, $a0, $a1
	}
	add $m4, $m2, 2
	shr $m1, $m6, 1
	and $m5, $m6, 1
	st32 $m5, $m11, $m15, 15                # 4-byte Folded Spill
	add $m1, $m1, $m5
	st32 $m1, $m11, $m15, 6                 # 4-byte Folded Spill
	setzi $m7, 1
	add $m5, $m2, 1
	add $m1, $m6, 1
	st32 $m1, $m11, $m15, 18                # 4-byte Folded Spill
	add $m1, $m6, -1
	st32 $m1, $m11, $m15, 17                # 4-byte Folded Spill
	st32 $m4, $m11, $m15, 8                 # 4-byte Folded Spill
	shl $m1, $m4, 3
	st32 $m1, $m11, $m15, 10                # 4-byte Folded Spill
	st32 $m2, $m11, $m15, 12                # 4-byte Folded Spill
	shl $m1, $m2, 3
	st32 $m1, $m11, $m15, 9                 # 4-byte Folded Spill
	mov	$m1, $m7
	st32 $m0, $m11, $m15, 21                # 4-byte Folded Spill
	st32 $m3, $m11, $m15, 13                # 4-byte Folded Spill
	st32 $m5, $m11, $m15, 14                # 4-byte Folded Spill
	st32 $m7, $m11, $m15, 11                # 4-byte Folded Spill
	bri .LBB3_2
.LBB3_3:                                #   in Loop: Header=BB3_2 Depth=1
	add $m1, $m1, 1
	cmpeq $m2, $m1, $m3
	brnz $m2, .LBB3_5
.LBB3_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_11 Depth 2
                                        #     Child Loop BB3_14 Depth 2
	cmpult $m2, $m7, $m5
	brz $m2, .LBB3_3
# %bb.9:                                #   in Loop: Header=BB3_2 Depth=1
	add $m3, $m1, -1
	ld32 $m4, $m0, $m15, 0
	add $m6, $m1, 1
	ld32 $m2, $m0, $m15, 2
	ld32 $m5, $m11, $m15, 15                # 4-byte Folded Reload
	st32 $m6, $m11, $m15, 16                # 4-byte Folded Spill
	add $m6, $m1, 1
	brnz $m5, .LBB3_10
# %bb.13:                               #   in Loop: Header=BB3_2 Depth=1
	ld32 $m5, $m11, $m15, 10                # 4-byte Folded Reload
	mul $m8, $m5, $m1
	ld32 $m1, $m11, $m15, 12                # 4-byte Folded Reload
	add $m9, $m4, 8
	add $m1, $m1, -1
	mul $m6, $m5, $m6
	add $m4, $m4, $m8
	mul $m8, $m5, $m3
	ld32 $m5, $m11, $m15, 9                 # 4-byte Folded Reload
	mul $m3, $m5, $m3
	add $m5, $m9, $m6
	add $m8, $m9, $m8
	add $m6, $m4, 16
	add $m9, $m2, $m3
.LBB3_14:                               #   Parent Loop BB3_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	ld32step $m10, $m15, $m8+=, 2
	add $m2, $m15, -16
	ld32 $m0, $m6, $m2, 0
	st32 $m0, $m11, $m15, 20                # 4-byte Folded Spill
	add $m2, $m15, -8
	ld32 $m4, $m6, $m2, 0
	ld32step $m7, $m15, $m6+=, 2
	ld32step $m3, $m15, $m5+=, 2
	ld32step $m2, $m15, $m9+=, 2
	st32 $m2, $m11, $m15, 19                # 4-byte Folded Spill
	ld32 $a1, $m10, $m15, 1
	ld32 $a2, $m3, $m15, 1
	{
		ld32 $a2, $m4, $m15, 1
		f32add $a1, $a1, $a2
	}
	{
		ld32 $a3, $m0, $m15, 1
		f32mul $a2, $a0, $a2
	}
	{
		ld32 $m0, $m11, $m15, 21
		f32add $a1, $a1, $a3
	}
	ld32 $a3, $m0, $m15, 7
	ld32 $a4, $m7, $m15, 1
	{
		ld32 $a4, $m4, $m15, 2
		f32add $a1, $a1, $a4
	}
	{
		ld32 $a4, $m4, $m15, 0
		f32add $a1, $a1, $a4
	}
	f32add $a1, $a1, $a4
	f32mul $a1, $a3, $a1
	f32add $a1, $a2, $a1
	st32 $a1, $m2, $m15, 0
	ld32 $m2, $m11, $m15, 22                # 4-byte Folded Reload
	ld32 $a1, $m10, $m15, $m2
	ld32 $m10, $m11, $m15, 17               # 4-byte Folded Reload
	ld32 $m2, $m11, $m15, 22                # 4-byte Folded Reload
	ld32 $a2, $m3, $m15, $m2
	{
		ld32 $a2, $m4, $m15, $m2
		f32add $a1, $a1, $a2
	}
	ld32 $m3, $m11, $m15, 20                # 4-byte Folded Reload
	{
		ld32 $a3, $m3, $m15, $m2
		f32mul $a2, $a0, $a2
	}
	{
		ld32 $a3, $m0, $m15, 7
		f32add $a1, $a1, $a3
	}
	ld32 $a4, $m7, $m15, $m2
	{
		ld32 $m0, $m11, $m15, 18
		f32add $a1, $a1, $a4
	}
	ld32 $a4, $m4, $m15, $m0
	{
		ld32 $a4, $m4, $m15, $m10
		f32add $a1, $a1, $a4
	}
	f32add $a1, $a1, $a4
	f32mul $a1, $a3, $a1
	{
		ld32 $m0, $m11, $m15, 19
		f32add $a1, $a2, $a1
	}
	st32 $a1, $m0, $m15, $m10
	brnzdec $m1, .LBB3_14
# %bb.15:                               #   in Loop: Header=BB3_2 Depth=1
	ld32 $m0, $m11, $m15, 21                # 4-byte Folded Reload
	bri .LBB3_12
.LBB3_10:                               #   in Loop: Header=BB3_2 Depth=1
	ld32 $m0, $m11, $m15, 10                # 4-byte Folded Reload
	mul $m8, $m0, $m1
	ld32 $m1, $m11, $m15, 12                # 4-byte Folded Reload
	add $m9, $m4, 8
	add $m1, $m1, -1
	mul $m6, $m0, $m6
	add $m4, $m4, $m8
	ld32 $m5, $m11, $m15, 9                 # 4-byte Folded Reload
	mul $m8, $m5, $m3
	mul $m0, $m0, $m3
	add $m3, $m9, $m6
	add $m6, $m2, $m8
	add $m8, $m4, 16
	add $m9, $m9, $m0
.LBB3_11:                               #   Parent Loop BB3_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	ld32step $m0, $m15, $m9+=, 2
	add $m2, $m15, -16
	ld32 $m2, $m8, $m2, 0
	add $m4, $m15, -8
	ld32 $m4, $m8, $m4, 0
	ld32step $m5, $m15, $m8+=, 2
	ld32step $m7, $m15, $m3+=, 2
	ld32step $m10, $m15, $m6+=, 2
	ld32 $a1, $m0, $m15, 1
	ld32 $a2, $m7, $m15, 1
	{
		ld32 $m0, $m11, $m15, 21
		f32add $a1, $a1, $a2
	}
	ld32 $a2, $m4, $m15, 1
	{
		ld32 $a3, $m2, $m15, 1
		f32mul $a2, $a0, $a2
	}
	{
		ld32 $a3, $m0, $m15, 7
		f32add $a1, $a1, $a3
	}
	ld32 $a4, $m5, $m15, 1
	{
		ld32 $a4, $m4, $m15, 2
		f32add $a1, $a1, $a4
	}
	{
		ld32 $a4, $m4, $m15, 0
		f32add $a1, $a1, $a4
	}
	f32add $a1, $a1, $a4
	f32mul $a1, $a3, $a1
	f32add $a1, $a2, $a1
	st32 $a1, $m10, $m15, 0
	brnzdec $m1, .LBB3_11
.LBB3_12:                               #   in Loop: Header=BB3_2 Depth=1
	ld32 $m3, $m11, $m15, 13                # 4-byte Folded Reload
	ld32 $m5, $m11, $m15, 14                # 4-byte Folded Reload
	ld32 $m7, $m11, $m15, 11                # 4-byte Folded Reload
	ld32 $m1, $m11, $m15, 16                # 4-byte Folded Reload
	cmpeq $m2, $m1, $m3
	brz $m2, .LBB3_2
.LBB3_5:
	ld32 $m1, $m11, $m15, 7                 # 4-byte Folded Reload
	brnz $m1, .LBB3_23
# %bb.6:
	{
		setzi $m1, 1
		mov	$a1, $a0
	}
	ld32 $m2, $m11, $m15, 6                 # 4-byte Folded Reload
	add $m4, $m2, -1
	st32 $m4, $m11, $m15, 14                # 4-byte Folded Spill
	cmpult $m4, $m1, $m5
	cmpult $m5, $m2, 2
	st64 $a0:1, $m11, $m15, 11              # 8-byte Folded Spill
	st32 $m4, $m11, $m15, 10                # 4-byte Folded Spill
	st32 $m5, $m11, $m15, 9                 # 4-byte Folded Spill
	bri .LBB3_7
.LBB3_8:                                #   in Loop: Header=BB3_7 Depth=1
	add $m7, $m1, 1
.LBB3_22:                               #   in Loop: Header=BB3_7 Depth=1
	cmpeq $m2, $m7, $m3
	mov	$m1, $m7
	brnz $m2, .LBB3_23
.LBB3_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_18 Depth 2
                                        #       Child Loop BB3_19 Depth 3
	brz $m4, .LBB3_8
# %bb.16:                               #   in Loop: Header=BB3_7 Depth=1
	add $m7, $m1, 1
	brnz $m5, .LBB3_22
# %bb.17:                               #   in Loop: Header=BB3_7 Depth=1
	ld32 $m3, $m11, $m15, 12                # 4-byte Folded Reload
	mov	$m6, $m3
	add $m2, $m1, -1
	ld32 $m4, $m11, $m15, 8                 # 4-byte Folded Reload
	mul $m1, $m1, $m4
	st32 $m1, $m11, $m15, 20                # 4-byte Folded Spill
	ld32 $m1, $m0, $m15, 0
	st32 $m1, $m11, $m15, 19                # 4-byte Folded Spill
	ld32 $m0, $m0, $m15, 2
	st32 $m0, $m11, $m15, 18                # 4-byte Folded Spill
	mul $m0, $m2, $m4
	st32 $m0, $m11, $m15, 17                # 4-byte Folded Spill
	mul $m0, $m2, $m3
	st32 $m0, $m11, $m15, 16                # 4-byte Folded Spill
	st32 $m7, $m11, $m15, 11                # 4-byte Folded Spill
	mul $m0, $m7, $m4
	st32 $m0, $m11, $m15, 15                # 4-byte Folded Spill
	setzi $m1, 1
.LBB3_18:                               #   Parent Loop BB3_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB3_19 Depth 3
	ld32 $m0, $m11, $m15, 17                # 4-byte Folded Reload
	add $m0, $m1, $m0
	shl $m0, $m0, 3
	ld32 $m8, $m11, $m15, 19                # 4-byte Folded Reload
	ld32 $m0, $m8, $m0, 0
	add $m5, $m0, 8
	add $m0, $m1, -1
	ld32 $m7, $m11, $m15, 20                # 4-byte Folded Reload
	add $m2, $m1, $m7
	shl $m2, $m2, 3
	ld32 $m2, $m8, $m2, 0
	add $m2, $m2, 8
	ld32 $m3, $m11, $m15, 15                # 4-byte Folded Reload
	add $m3, $m1, $m3
	add $m1, $m1, 1
	shl $m3, $m3, 3
	ld32 $m3, $m8, $m3, 0
	add $m9, $m3, 8
	add $m3, $m0, $m7
	ld32 $m4, $m11, $m15, 16                # 4-byte Folded Reload
	add $m0, $m0, $m4
	shl $m3, $m3, 3
	shl $m0, $m0, 3
	ld32 $m3, $m8, $m3, 0
	ld32 $m4, $m11, $m15, 18                # 4-byte Folded Reload
	ld32 $m0, $m4, $m0, 0
	add $m10, $m0, 4
	add $m4, $m3, 8
	add $m0, $m1, $m7
	shl $m0, $m0, 3
	ld32 $m0, $m8, $m0, 0
	add $m3, $m0, 8
	ld32 $m7, $m11, $m15, 14                # 4-byte Folded Reload
	add $m7, $m7, -1
	ld32 $m8, $m11, $m15, 21                # 4-byte Folded Reload
.LBB3_19:                               #   Parent Loop BB3_7 Depth=1
                                        #     Parent Loop BB3_18 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add $m0, $m15, -4
	ld32 $a2, $m2, $m0, 0
	ld64step $a4:5, $m15, $m2+=, 1
	{
		ld32 $a3, $m2, $m15, 0
		f32add $a2, $a2, $a5
	}
	{
		ld32 $a6, $m8, $m15, 7
		f32add $a3, $a4, $a3
	}
	ld64 $a0:1, $m11, $m15, 11              # 8-byte Folded Reload
	{
		ld64step $a0:1, $m15, $m5+=, 1
		f32v2mul $a4:5, $a0:1, $a4:5
	}
	{
		ld64step $a2:3, $m15, $m9+=, 1
		f32v2add $a0:1, $a0:1, $a2:3
	}
	{
		ld64step $a2:3, $m15, $m4+=, 1
		f32v2add $a0:1, $a2:3, $a0:1
	}
	{
		ld64step $a2:3, $m15, $m3+=, 1
		f32v2add $a0:1, $a2:3, $a0:1
	}
	f32v2add $a0:1, $a2:3, $a0:1
	f32v2mul $a0:1, $a6:B, $a0:1
	f32v2add $a0:1, $a4:5, $a0:1
	st32 $a1, $m10, $m15, 1
	st32step $a0, $m15, $m10+=, 2
	brnzdec $m7, .LBB3_19
# %bb.20:                               #   in Loop: Header=BB3_18 Depth=2
	add $m6, $m6, -1
	brnz $m6, .LBB3_18
# %bb.21:                               #   in Loop: Header=BB3_7 Depth=1
	ld32 $m0, $m11, $m15, 21                # 4-byte Folded Reload
	ld32 $m3, $m11, $m15, 13                # 4-byte Folded Reload
	ld32 $m4, $m11, $m15, 10                # 4-byte Folded Reload
	ld32 $m5, $m11, $m15, 9                 # 4-byte Folded Reload
	ld32 $m7, $m11, $m15, 11                # 4-byte Folded Reload
	bri .LBB3_22
.LBB3_23:
	setzi $m0, 1
	ld32 $m7, $m11, $m15, 1                 # 4-byte Folded Reload
	ld32 $a6, $m11, $m15, 2                 # 4-byte Folded Reload
	ld32 $m10, $m11, $m15, 3                # 4-byte Folded Reload
	ld32 $m9, $m11, $m15, 4                 # 4-byte Folded Reload
	ld32 $m8, $m11, $m15, 5                 # 4-byte Folded Reload
	add $m11, $m11, 96
	br $m10
.Lfunc_end3:
	.size	_ZN21HeatEquationOptimized7computeEv, .Lfunc_end3-_ZN21HeatEquationOptimized7computeEv
	.section	.stack_sizes,"Go",@progbits,_ZN21HeatEquationOptimized7computeEv,comdat,.text._ZN21HeatEquationOptimized7computeEv
	.long	.Lfunc_begin3
	.byte	96
	.section	.text._ZN21HeatEquationOptimized7computeEv,"axG",@progbits,_ZN21HeatEquationOptimized7computeEv,comdat
                                        # -- End function
	.ident	"clang version 13.0.0 (ssh://git@phabricator.sourcevertex.net/diffusion/LLVMPROJECT/llvm-project.git 6eeafc711ed2cca929acfbe560c20764774f9a28)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
