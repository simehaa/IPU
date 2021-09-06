	.text
	.allow_optimizations
	.file	"3"
	.section	.text.__runCodelet_HeatEquation,"ax",@progbits
	.globl	__runCodelet_HeatEquation       # -- Begin function __runCodelet_HeatEquation
	.p2align	2
	.type	__runCodelet_HeatEquation,@function
__runCodelet_HeatEquation:              # @__runCodelet_HeatEquation
.Lfunc_begin0:
# %bb.0:
	mov	$m11, $m12
	mov	$m0, $m13
	call $m10, _ZN12HeatEquation7computeEv
	exitnz $m0
.Lfunc_end0:
	.size	__runCodelet_HeatEquation, .Lfunc_end0-__runCodelet_HeatEquation
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_HeatEquation
	.long	.Lfunc_begin0
	.byte	0
	.section	.text.__runCodelet_HeatEquation,"ax",@progbits
                                        # -- End function
	.section	.text._ZN12HeatEquation7computeEv,"axG",@progbits,_ZN12HeatEquation7computeEv,comdat
	.weak	_ZN12HeatEquation7computeEv     # -- Begin function _ZN12HeatEquation7computeEv
	.p2align	2
	.type	_ZN12HeatEquation7computeEv,@function
_ZN12HeatEquation7computeEv:            # @_ZN12HeatEquation7computeEv
.Lfunc_begin1:
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
	brnz $m1, .LBB1_23
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
	bri .LBB1_2
.LBB1_3:                                #   in Loop: Header=BB1_2 Depth=1
	add $m1, $m1, 1
	cmpeq $m2, $m1, $m3
	brnz $m2, .LBB1_5
.LBB1_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_11 Depth 2
                                        #     Child Loop BB1_14 Depth 2
	cmpult $m2, $m7, $m5
	brz $m2, .LBB1_3
# %bb.9:                                #   in Loop: Header=BB1_2 Depth=1
	add $m3, $m1, -1
	ld32 $m4, $m0, $m15, 0
	add $m6, $m1, 1
	ld32 $m2, $m0, $m15, 2
	ld32 $m5, $m11, $m15, 15                # 4-byte Folded Reload
	st32 $m6, $m11, $m15, 16                # 4-byte Folded Spill
	add $m6, $m1, 1
	brnz $m5, .LBB1_10
# %bb.13:                               #   in Loop: Header=BB1_2 Depth=1
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
.LBB1_14:                               #   Parent Loop BB1_2 Depth=1
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
	brnzdec $m1, .LBB1_14
# %bb.15:                               #   in Loop: Header=BB1_2 Depth=1
	ld32 $m0, $m11, $m15, 21                # 4-byte Folded Reload
	bri .LBB1_12
.LBB1_10:                               #   in Loop: Header=BB1_2 Depth=1
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
.LBB1_11:                               #   Parent Loop BB1_2 Depth=1
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
	brnzdec $m1, .LBB1_11
.LBB1_12:                               #   in Loop: Header=BB1_2 Depth=1
	ld32 $m3, $m11, $m15, 13                # 4-byte Folded Reload
	ld32 $m5, $m11, $m15, 14                # 4-byte Folded Reload
	ld32 $m7, $m11, $m15, 11                # 4-byte Folded Reload
	ld32 $m1, $m11, $m15, 16                # 4-byte Folded Reload
	cmpeq $m2, $m1, $m3
	brz $m2, .LBB1_2
.LBB1_5:
	ld32 $m1, $m11, $m15, 7                 # 4-byte Folded Reload
	brnz $m1, .LBB1_23
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
	bri .LBB1_7
.LBB1_8:                                #   in Loop: Header=BB1_7 Depth=1
	add $m7, $m1, 1
.LBB1_22:                               #   in Loop: Header=BB1_7 Depth=1
	cmpeq $m2, $m7, $m3
	mov	$m1, $m7
	brnz $m2, .LBB1_23
.LBB1_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_18 Depth 2
                                        #       Child Loop BB1_19 Depth 3
	brz $m4, .LBB1_8
# %bb.16:                               #   in Loop: Header=BB1_7 Depth=1
	add $m7, $m1, 1
	brnz $m5, .LBB1_22
# %bb.17:                               #   in Loop: Header=BB1_7 Depth=1
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
.LBB1_18:                               #   Parent Loop BB1_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_19 Depth 3
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
.LBB1_19:                               #   Parent Loop BB1_7 Depth=1
                                        #     Parent Loop BB1_18 Depth=2
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
	brnzdec $m7, .LBB1_19
# %bb.20:                               #   in Loop: Header=BB1_18 Depth=2
	add $m6, $m6, -1
	brnz $m6, .LBB1_18
# %bb.21:                               #   in Loop: Header=BB1_7 Depth=1
	ld32 $m0, $m11, $m15, 21                # 4-byte Folded Reload
	ld32 $m3, $m11, $m15, 13                # 4-byte Folded Reload
	ld32 $m4, $m11, $m15, 10                # 4-byte Folded Reload
	ld32 $m5, $m11, $m15, 9                 # 4-byte Folded Reload
	ld32 $m7, $m11, $m15, 11                # 4-byte Folded Reload
	bri .LBB1_22
.LBB1_23:
	setzi $m0, 1
	ld32 $m7, $m11, $m15, 1                 # 4-byte Folded Reload
	ld32 $a6, $m11, $m15, 2                 # 4-byte Folded Reload
	ld32 $m10, $m11, $m15, 3                # 4-byte Folded Reload
	ld32 $m9, $m11, $m15, 4                 # 4-byte Folded Reload
	ld32 $m8, $m11, $m15, 5                 # 4-byte Folded Reload
	add $m11, $m11, 96
	br $m10
.Lfunc_end1:
	.size	_ZN12HeatEquation7computeEv, .Lfunc_end1-_ZN12HeatEquation7computeEv
	.section	.stack_sizes,"Go",@progbits,_ZN12HeatEquation7computeEv,comdat,.text._ZN12HeatEquation7computeEv
	.long	.Lfunc_begin1
	.byte	96
	.section	.text._ZN12HeatEquation7computeEv,"axG",@progbits,_ZN12HeatEquation7computeEv,comdat
                                        # -- End function
	.ident	"clang version 13.0.0 (ssh://git@phabricator.sourcevertex.net/diffusion/LLVMPROJECT/llvm-project.git a39a2acfae1c9741d5aa0363a690339a63142bea)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
