	.text
	.allow_optimizations
	.file	"codelet-mow16mf0.cpp"
	.section	.text.__runCodelet_HeatEquationSimple,"ax",@progbits
	.globl	__runCodelet_HeatEquationSimple # -- Begin function __runCodelet_HeatEquationSimple
	.p2align	2
	.type	__runCodelet_HeatEquationSimple,@function
__runCodelet_HeatEquationSimple:        # @__runCodelet_HeatEquationSimple
.Lfunc_begin0:
# %bb.0:
	setzi $m6, 80
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	add $m11, $m12, -80
	ld32 $m0, $m13, $m15, 4
	st32 $m0, $m11, $m15, 10                # 4-byte Folded Spill
	cmpslt $m0, $m0, 1
	brnz $m0, .LBB0_16
# %bb.1:
	ld32 $m0, $m13, $m15, 5
	st32 $m0, $m11, $m15, 11                # 4-byte Folded Spill
	cmpslt $m0, $m0, 1
	brnz $m0, .LBB0_16
# %bb.2:
	{
		ld32 $a1, $m13, $m15, 6
		or $a0, $a15, 3229614080
	}
	f32mul $a0, $a1, $a0
	or $a1, $a15, 1065353216
	{
		add $m2, $m13, 24
		f32add $a6, $a0, $a1
	}
	{
		ld32 $m0, $m11, $m15, 11
		mov	$a7, $a6
	}
	add $m1, $m0, 1
	st32 $m1, $m11, $m15, 6                 # 4-byte Folded Spill
	or $m1, $m0, 1
	st32 $m1, $m11, $m15, 3                 # 4-byte Folded Spill
	add $m1, $m13, 25
	st32 $m1, $m11, $m15, 5                 # 4-byte Folded Spill
	ld32 $m1, $m13, $m15, 2
	st32 $m1, $m11, $m15, 9                 # 4-byte Folded Spill
	shl $m1, $m0, 2
	add $m3, $m1, 8
	st32 $m3, $m11, $m15, 4                 # 4-byte Folded Spill
	st32 $m1, $m11, $m15, 7                 # 4-byte Folded Spill
	add $m1, $m1, 4
	st32 $m1, $m11, $m15, 12                # 4-byte Folded Spill
	ld32 $m1, $m13, $m15, 0
	ld32 $m3, $m1, $m15, 2
	st32 $m3, $m11, $m15, 14                # 4-byte Folded Spill
	st32 $m1, $m11, $m15, 8                 # 4-byte Folded Spill
	ld32 $m8, $m1, $m15, 0
	andc $m0, $m0, 1
	st32 $m0, $m11, $m15, 2                 # 4-byte Folded Spill
	add $m0, $m0, -2
	shr $m0, $m0, 1
	add $m0, $m0, 1
	st32 $m0, $m11, $m15, 1                 # 4-byte Folded Spill
	setzi $m4, 1
	st32 $m4, $m11, $m15, 15                # 4-byte Folded Spill
	st64 $a6:7, $m11, $m15, 8               # 8-byte Folded Spill
	bri .LBB0_3
.LBB0_15:                               #   in Loop: Header=BB0_3 Depth=1
	ld32 $m0, $m11, $m15, 10                # 4-byte Folded Reload
	cmpslt $m0, $m4, $m0
	mov	$m8, $m3
	ld32 $m4, $m11, $m15, 13                # 4-byte Folded Reload
	brz $m0, .LBB0_16
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_13 Depth 2
                                        #     Child Loop BB0_5 Depth 2
	ld32 $m3, $m11, $m15, 14                # 4-byte Folded Reload
	shl $m0, $m4, 3
	ld32 $m1, $m11, $m15, 9                 # 4-byte Folded Reload
	add $m0, $m0, $m1
	add $m1, $m15, -8
	ld32 $m7, $m0, $m1, 0
	add $m0, $m4, 1
	st32 $m0, $m11, $m15, 13                # 4-byte Folded Spill
	shl $m0, $m0, 3
	ld32 $m1, $m11, $m15, 8                 # 4-byte Folded Reload
	ld32 $m0, $m1, $m0, 0
	st32 $m0, $m11, $m15, 14                # 4-byte Folded Spill
	ld32 $m0, $m11, $m15, 11                # 4-byte Folded Reload
	ld32 $m5, $m11, $m15, 15                # 4-byte Folded Reload
	cmpult $m0, $m5, $m0
	brnz $m0, .LBB0_6
.LBB0_4:                                #   in Loop: Header=BB0_3 Depth=1
	shl $m9, $m5, 2
	ld32 $m0, $m11, $m15, 14                # 4-byte Folded Reload
	add $m0, $m0, $m9
	add $m1, $m3, $m9
	add $m6, $m8, $m9
	ld32 $m8, $m11, $m15, 6                 # 4-byte Folded Reload
	sub $m8, $m8, $m5
	add $m8, $m8, -1
	add $m5, $m7, $m9
	add $m7, $m5, -4
.LBB0_5:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add $m5, $m15, -4
	ld32 $a2, $m1, $m5, 0
	ld32step $a3, $m15, $m0+=, 1
	ld32step $a4, $m15, $m6+=, 1
	f32add $a3, $a4, $a3
	{
		ld32step $a3, $m15, $m1+=, 1
		f32add $a2, $a3, $a2
	}
	ld32 $a4, $m1, $m15, 0
	{
		ld32 $a4, $m2, $m15, 0
		f32add $a2, $a2, $a4
	}
	f32mul $a2, $a4, $a2
	f32mul $a3, $a6, $a3
	f32add $a2, $a3, $a2
	st32step $a2, $m15, $m7+=, 1
	brnzdec $m8, .LBB0_5
	bri .LBB0_15
.LBB0_6:                                #   in Loop: Header=BB0_3 Depth=1
	ld32 $m0, $m11, $m15, 7                 # 4-byte Folded Reload
	add $m0, $m7, $m0
	add $m1, $m3, 4
	cmpult $m1, $m1, $m0
	ld32 $m5, $m11, $m15, 12                # 4-byte Folded Reload
	add $m5, $m3, $m5
	cmpult $m5, $m7, $m5
	and $m1, $m5, $m1
	ld32 $m5, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m1, .LBB0_4
# %bb.7:                                #   in Loop: Header=BB0_3 Depth=1
	ld32 $m1, $m11, $m15, 5                 # 4-byte Folded Reload
	cmpult $m1, $m7, $m1
	cmpult $m5, $m2, $m0
	and $m1, $m1, $m5
	ld32 $m5, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m1, .LBB0_4
# %bb.8:                                #   in Loop: Header=BB0_3 Depth=1
	add $m10, $m8, 4
	cmpult $m1, $m10, $m0
	ld32 $m5, $m11, $m15, 12                # 4-byte Folded Reload
	add $m5, $m8, $m5
	cmpult $m5, $m7, $m5
	and $m1, $m5, $m1
	ld32 $m5, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m1, .LBB0_4
# %bb.9:                                #   in Loop: Header=BB0_3 Depth=1
	ld32 $m9, $m11, $m15, 14                # 4-byte Folded Reload
	add $m1, $m9, 4
	cmpult $m5, $m1, $m0
	ld32 $m6, $m11, $m15, 12                # 4-byte Folded Reload
	add $m6, $m9, $m6
	cmpult $m6, $m7, $m6
	and $m6, $m6, $m5
	ld32 $m5, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m6, .LBB0_4
# %bb.10:                               #   in Loop: Header=BB0_3 Depth=1
	cmpult $m5, $m3, $m0
	ld32 $m6, $m11, $m15, 7                 # 4-byte Folded Reload
	add $m6, $m3, $m6
	cmpult $m6, $m7, $m6
	and $m6, $m6, $m5
	ld32 $m5, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m6, .LBB0_4
# %bb.11:                               #   in Loop: Header=BB0_3 Depth=1
	add $m9, $m3, 8
	cmpult $m0, $m9, $m0
	ld32 $m5, $m11, $m15, 4                 # 4-byte Folded Reload
	add $m5, $m3, $m5
	cmpult $m5, $m7, $m5
	and $m0, $m5, $m0
	ld32 $m5, $m11, $m15, 15                # 4-byte Folded Reload
	brnz $m0, .LBB0_4
# %bb.12:                               #   in Loop: Header=BB0_3 Depth=1
	ld32 $a0, $m2, $m15, 0
	mov	$a1, $a0
	st64 $a0:1, $m11, $m15, 9               # 8-byte Folded Spill
	ld32 $m6, $m11, $m15, 1                 # 4-byte Folded Reload
	add $m6, $m6, -1
	mov	$m0, $m7
.LBB0_13:                               #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add $m5, $m15, -4
	ld32 $a5, $m9, $m5, 0
	add $m5, $m15, -8
	ld32 $a4, $m9, $m5, 0
	ld32 $a7, $m1, $m15, 1
	ld32step $a6, $m15, $m1+=, 2
	ld32 $a3, $m10, $m15, 1
	ld32step $a2, $m15, $m10+=, 2
	f32v2add $a2:3, $a2:3, $a6:7
	{
		ld32 $a7, $m9, $m15, 1
		f32v2add $a2:3, $a2:3, $a4:5
	}
	ld32step $a6, $m15, $m9+=, 2
	{
		ld64 $a0:1, $m11, $m15, 9
		f32v2add $a2:3, $a2:3, $a6:7
	}
	f32v2mul $a2:3, $a0:1, $a2:3
	mov	$a4, $a5
	{
		ld64 $a6:7, $m11, $m15, 8
		mov	$a5, $a6
	}
	f32v2mul $a4:5, $a6:7, $a4:5
	f32v2add $a2:3, $a4:5, $a2:3
	st32 $a3, $m0, $m15, 1
	st32step $a2, $m15, $m0+=, 2
	brnzdec $m6, .LBB0_13
# %bb.14:                               #   in Loop: Header=BB0_3 Depth=1
	ld32 $m0, $m11, $m15, 11                # 4-byte Folded Reload
	ld32 $m1, $m11, $m15, 2                 # 4-byte Folded Reload
	cmpne $m0, $m0, $m1
	ld32 $m5, $m11, $m15, 3                 # 4-byte Folded Reload
	brz $m0, .LBB0_15
	bri .LBB0_4
.LBB0_16:
	exitz $m15
.Lfunc_end0:
	.size	__runCodelet_HeatEquationSimple, .Lfunc_end0-__runCodelet_HeatEquationSimple
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_HeatEquationSimple
	.long	.Lfunc_begin0
	.byte	80
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
	add $m11, $m11, -40
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st32 $m8, $m11, $m15, 7                 # 4-byte Folded Spill
	st32 $m9, $m11, $m15, 6                 # 4-byte Folded Spill
	st32 $m10, $m11, $m15, 5                # 4-byte Folded Spill
	st64 $a6:7, $m11, $m15, 1               # 8-byte Folded Spill
	st32 $m7, $m11, $m15, 1                 # 4-byte Folded Spill
	ld32 $m1, $m0, $m15, 4
	st32 $m1, $m11, $m15, 9                 # 4-byte Folded Spill
	cmpslt $m7, $m1, 1
	brnz $m7, .LBB2_11
# %bb.1:
	{
		ld32 $m2, $m0, $m15, 5
		or $a0, $a15, 3229614080
	}
	ld32 $a1, $m0, $m15, 6
	f32mul $a0, $a1, $a0
	or $a2, $a15, 1065353216
	{
		ld32 $m4, $m0, $m15, 0
		f32add $a0, $a0, $a2
	}
	ld32 $m3, $m4, $m15, 2
	ld32 $m1, $m11, $m15, 9                 # 4-byte Folded Reload
	add $m5, $m1, -1
	ld32 $m6, $m0, $m15, 2
	ld32step $m9, $m15, $m4+=, 4
	and $m1, $m2, 1
	st32 $m7, $m11, $m15, 8                 # 4-byte Folded Spill
	brnz $m1, .LBB2_12
# %bb.2:
	add $m8, $m2, 1
	add $m10, $m2, -1
.LBB2_3:                                # =>This Inner Loop Header: Depth=1
	mov	$m7, $m3
	ld32 $a2, $m7, $m15, 1
	{
		ld32 $a3, $m7, $m15, 0
		f32mul $a2, $a0, $a2
	}
	ld32 $a4, $m9, $m15, 1
	{
		ld32 $a4, $m7, $m15, 2
		f32add $a3, $a4, $a3
	}
	{
		ld32 $m3, $m4, $m15, 0
		f32add $a3, $a3, $a4
	}
	ld32 $a4, $m3, $m15, 1
	f32add $a3, $a3, $a4
	f32mul $a1, $a1, $a3
	{
		ld32 $m1, $m6, $m15, 0
		f32add $a1, $a2, $a1
	}
	st32 $a1, $m1, $m15, 0
	ld32 $a1, $m7, $m15, $m10
	ld32 $a2, $m9, $m15, $m2
	{
		ld32 $a2, $m7, $m15, $m8
		f32add $a1, $a2, $a1
	}
	{
		ld32 $a2, $m3, $m15, $m2
		f32add $a1, $a1, $a2
	}
	{
		ld32 $a2, $m0, $m15, 6
		f32add $a1, $a1, $a2
	}
	{
		ld32 $a2, $m7, $m15, $m2
		f32mul $a1, $a2, $a1
	}
	f32mul $a2, $a0, $a2
	f32add $a1, $a2, $a1
	st32 $a1, $m1, $m15, $m10
	brz $m5, .LBB2_5
# %bb.4:                                #   in Loop: Header=BB2_3 Depth=1
	add $m6, $m6, 8
	add $m4, $m4, 8
	add $m5, $m5, -1
	ld32 $a1, $m0, $m15, 6
	mov	$m9, $m7
	bri .LBB2_3
.LBB2_12:                               # =>This Inner Loop Header: Depth=1
	mov	$m1, $m3
	ld32 $a2, $m1, $m15, 1
	{
		ld32 $a3, $m1, $m15, 0
		f32mul $a2, $a0, $a2
	}
	ld32 $a4, $m9, $m15, 1
	{
		ld32 $a4, $m1, $m15, 2
		f32add $a3, $a4, $a3
	}
	{
		ld32 $m3, $m4, $m15, 0
		f32add $a3, $a3, $a4
	}
	ld32 $a4, $m3, $m15, 1
	f32add $a3, $a3, $a4
	f32mul $a1, $a1, $a3
	{
		ld32 $m7, $m6, $m15, 0
		f32add $a1, $a2, $a1
	}
	st32 $a1, $m7, $m15, 0
	brz $m5, .LBB2_5
# %bb.13:                               #   in Loop: Header=BB2_12 Depth=1
	add $m6, $m6, 8
	add $m4, $m4, 8
	add $m5, $m5, -1
	ld32 $a1, $m0, $m15, 6
	mov	$m9, $m1
	bri .LBB2_12
.LBB2_5:
	ld32 $m1, $m11, $m15, 8                 # 4-byte Folded Reload
	brnz $m1, .LBB2_11
# %bb.6:
	shr $m1, $m2, 31
	add $m1, $m2, $m1
	andc $m3, $m1, 1
	sub $m3, $m2, $m3
	shrs $m1, $m1, 1
	add $m4, $m1, $m3
	setzi $m3, 1
	cmpslt $m4, $m3, $m4
	brz $m4, .LBB2_11
# %bb.7:
	{
		ld32 $m4, $m0, $m15, 2
		mov	$a1, $a0
	}
	st32 $m4, $m11, $m15, 8                 # 4-byte Folded Spill
	ld32 $m5, $m0, $m15, 0
	xnor $m1, $m1, $m15
	add $m1, $m1, $m2
	ld32 $m6, $m5, $m15, 2
	ld32 $m2, $m5, $m15, 0
.LBB2_8:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_9 Depth 2
	mov	$m7, $m6
	add $m8, $m3, 1
	shl $m4, $m8, 3
	ld32 $m6, $m5, $m4, 0
	shl $m4, $m3, 3
	ld32 $m9, $m11, $m15, 8                 # 4-byte Folded Reload
	add $m4, $m4, $m9
	add $m9, $m15, -8
	ld32 $m4, $m4, $m9, 0
	add $m9, $m4, 4
	add $m10, $m2, 8
	mov	$m2, $m1
	add $m2, $m2, -1
	mov	$m4, $m15
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
# %bb.10:                               #   in Loop: Header=BB2_8 Depth=1
	ld32 $m2, $m11, $m15, 9                 # 4-byte Folded Reload
	cmpeq $m4, $m3, $m2
	mov	$m2, $m7
	mov	$m3, $m8
	brz $m4, .LBB2_8
.LBB2_11:
	setzi $m0, 1
	ld32 $m7, $m11, $m15, 1                 # 4-byte Folded Reload
	ld64 $a6:7, $m11, $m15, 1               # 8-byte Folded Reload
	ld32 $m10, $m11, $m15, 5                # 4-byte Folded Reload
	ld32 $m9, $m11, $m15, 6                 # 4-byte Folded Reload
	ld32 $m8, $m11, $m15, 7                 # 4-byte Folded Reload
	add $m11, $m11, 40
	br $m10
.Lfunc_end2:
	.size	_ZN21HeatEquationOptimized7computeEv, .Lfunc_end2-_ZN21HeatEquationOptimized7computeEv
	.section	.stack_sizes,"Go",@progbits,_ZN21HeatEquationOptimized7computeEv,comdat,.text._ZN21HeatEquationOptimized7computeEv
	.long	.Lfunc_begin2
	.byte	40
	.section	.text._ZN21HeatEquationOptimized7computeEv,"axG",@progbits,_ZN21HeatEquationOptimized7computeEv,comdat
                                        # -- End function
	.ident	"clang version 11.0.0 (git@phabricator.sourcevertex.net:diffusion/LLVMPROJECT/llvm-project.git 75a6ae69cf51c0c33aa5d3311d1aecd27418292c)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
