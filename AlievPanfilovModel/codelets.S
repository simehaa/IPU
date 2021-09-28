	.text
	.allow_optimizations
	.file	"3"
	.section	.text.__runCodelet_AlievPanfilov,"ax",@progbits
	.globl	__runCodelet_AlievPanfilov      # -- Begin function __runCodelet_AlievPanfilov
	.p2align	2
	.type	__runCodelet_AlievPanfilov,@function
__runCodelet_AlievPanfilov:             # @__runCodelet_AlievPanfilov
.Lfunc_begin0:
# %bb.0:
	mov	$m11, $m12
	mov	$m0, $m13
	call $m10, _ZN13AlievPanfilov7computeEv
	exitnz $m0
.Lfunc_end0:
	.size	__runCodelet_AlievPanfilov, .Lfunc_end0-__runCodelet_AlievPanfilov
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_AlievPanfilov
	.long	.Lfunc_begin0
	.byte	0
	.section	.text.__runCodelet_AlievPanfilov,"ax",@progbits
                                        # -- End function
	.section	.text._ZN13AlievPanfilov7computeEv,"axG",@progbits,_ZN13AlievPanfilov7computeEv,comdat
	.weak	_ZN13AlievPanfilov7computeEv    # -- Begin function _ZN13AlievPanfilov7computeEv
	.p2align	2
	.type	_ZN13AlievPanfilov7computeEv,@function
_ZN13AlievPanfilov7computeEv:           # @_ZN13AlievPanfilov7computeEv
.Lfunc_begin1:
# %bb.0:
	add $m11, $m11, -80
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st32 $m8, $m11, $m15, 6                 # 4-byte Folded Spill
	st32 $m9, $m11, $m15, 5                 # 4-byte Folded Spill
	st32 $m10, $m11, $m15, 4                # 4-byte Folded Spill
	st64 $a6:7, $m11, $m15, 1               # 8-byte Folded Spill
	st32 $m7, $m11, $m15, 1                 # 4-byte Folded Spill
	ld32 $m1, $m0, $m15, 8
	cmpslt $m2, $m1, 1
	brnz $m2, .LBB1_6
# %bb.1:
	ld32 $m2, $m0, $m15, 9
	st32 $m2, $m11, $m15, 11                # 4-byte Folded Spill
	cmpslt $m2, $m2, 1
	brnz $m2, .LBB1_6
# %bb.2:
	mov	$m4, $m15
	{
		ld32 $a0, $m0, $m15, 14
		or $a2, $a15, 3212836864
	}
	{
		ld32 $m2, $m0, $m15, 0
		f32mul $a0, $a0, $a0
	}
	ld32 $m3, $m0, $m15, 4
	ld32 $a1, $m0, $m15, 10
	f32div $a0, $a1, $a0
	st32 $a0, $m11, $m15, 17                # 4-byte Folded Spill
	ld32 $a1, $m0, $m15, 11
	{
		ld32 $a3, $m0, $m15, 18
		f32sub $a4, $a15, $a1
	}
	st32 $m2, $m11, $m15, 10                # 4-byte Folded Spill
	{
		ld32 $m5, $m2, $m15, 2
		f32add $a2, $a3, $a2
	}
	ld32 $m2, $m0, $m15, 2
	st32 $m2, $m11, $m15, 8                 # 4-byte Folded Spill
	ld32 $m2, $m0, $m15, 6
	st32 $m2, $m11, $m15, 7                 # 4-byte Folded Spill
	st32 $m3, $m11, $m15, 9                 # 4-byte Folded Spill
	ld32 $m2, $m3, $m15, 0
	st32 $a4, $m11, $m15, 16                # 4-byte Folded Spill
	add $m9, $m15, -4
.LBB1_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_4 Depth 2
	st32 $m2, $m11, $m15, 19                # 4-byte Folded Spill
	st32 $m1, $m11, $m15, 15                # 4-byte Folded Spill
	mov	$m10, $m15
	mov	$m2, $m4
	mov	$m3, $m5
	add $m4, $m3, 4
	shl $m5, $m2, 3
	add $m1, $m2, 1
	st32 $m1, $m11, $m15, 14                # 4-byte Folded Spill
	shl $m2, $m1, 3
	ld32 $m1, $m11, $m15, 9                 # 4-byte Folded Reload
	ld32 $m3, $m1, $m2, 0
	ld32 $m7, $m11, $m15, 11                # 4-byte Folded Reload
	ld32 $m1, $m11, $m15, 10                # 4-byte Folded Reload
	ld32 $m6, $m1, $m5, 4
	ld32 $m2, $m1, $m5, 0
	add $m8, $m2, 4
	ld32 $m1, $m11, $m15, 8                 # 4-byte Folded Reload
	ld32 $m1, $m1, $m5, 0
	st32 $m1, $m11, $m15, 18                # 4-byte Folded Spill
	ld32 $m1, $m11, $m15, 7                 # 4-byte Folded Reload
	ld32 $m5, $m1, $m5, 0
	add $m7, $m7, -1
	st32 $m6, $m11, $m15, 13                # 4-byte Folded Spill
	add $m6, $m6, 4
	st32 $m3, $m11, $m15, 12                # 4-byte Folded Spill
	add $m2, $m3, 4
.LBB1_4:                                #   Parent Loop BB1_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	{
		ld32 $a3, $m4, $m10, 0
		or $a4, $a15, 3229614080
	}
	{
		ld32 $a5, $m6, $m10, 0
		f32mul $a4, $a3, $a4
	}
	{
		ld32 $a5, $m8, $m10, 0
		f32add $a4, $a5, $a4
	}
	{
		ld32 $a5, $m4, $m10, 1
		f32add $a4, $a4, $a5
	}
	{
		ld32 $a6, $m0, $m15, 16
		f32add $a4, $a4, $a5
	}
	{
		ld32 $a5, $m0, $m15, 17
		f32mul $a6, $a3, $a6
	}
	{
		ld32 $a7, $m2, $m10, 0
		f32sub $a5, $a3, $a5
	}
	{
		ld32 $a6, $m0, $m15, 15
		f32mul $a5, $a6, $a5
	}
	{
		add $m1, $m4, $m10
		f32mul $a7, $a3, $a7
	}
	or $a0, $a15, 3212836864
	{
		ld32 $a1, $m1, $m9, 0
		f32add $a0, $a3, $a0
	}
	{
		add $m1, $m8, $m10
		f32add $a1, $a4, $a1
	}
	ld32 $a4, $m11, $m15, 17                # 4-byte Folded Reload
	f32mul $a1, $a4, $a1
	f32mul $a0, $a0, $a5
	f32sub $a0, $a1, $a0
	f32sub $a0, $a0, $a7
	f32mul $a0, $a0, $a6
	{
		ld32 $m3, $m11, $m15, 18
		f32add $a0, $a3, $a0
	}
	st32 $a0, $m3, $m10, 0
	ld32 $a0, $m0, $m15, 12
	ld32 $m3, $m11, $m15, 19                # 4-byte Folded Reload
	ld32 $a1, $m3, $m10, 0
	{
		ld32 $a3, $m0, $m15, 13
		f32mul $a0, $a0, $a1
	}
	ld32 $a4, $m1, $m9, 0
	{
		ld32 $a5, $m0, $m15, 16
		f32add $a3, $a3, $a4
	}
	{
		ld32 $a6, $m0, $m15, 15
		f32mul $a5, $a4, $a5
	}
	f32sub $a4, $a4, $a2
	f32div $a0, $a0, $a3
	{
		ld32 $a4, $m11, $m15, 16
		f32mul $a3, $a4, $a5
	}
	f32sub $a0, $a4, $a0
	f32add $a3, $a1, $a3
	f32mul $a0, $a0, $a3
	f32mul $a0, $a0, $a6
	f32add $a0, $a1, $a0
	st32 $a0, $m5, $m10, 0
	add $m10, $m10, 4
	brnzdec $m7, .LBB1_4
# %bb.5:                                #   in Loop: Header=BB1_3 Depth=1
	ld32 $m1, $m11, $m15, 15                # 4-byte Folded Reload
	add $m1, $m1, -1
	ld32 $m2, $m11, $m15, 12                # 4-byte Folded Reload
	ld32 $m4, $m11, $m15, 14                # 4-byte Folded Reload
	ld32 $m5, $m11, $m15, 13                # 4-byte Folded Reload
	brnz $m1, .LBB1_3
.LBB1_6:
	setzi $m0, 1
	ld32 $m7, $m11, $m15, 1                 # 4-byte Folded Reload
	ld64 $a6:7, $m11, $m15, 1               # 8-byte Folded Reload
	ld32 $m10, $m11, $m15, 4                # 4-byte Folded Reload
	ld32 $m9, $m11, $m15, 5                 # 4-byte Folded Reload
	ld32 $m8, $m11, $m15, 6                 # 4-byte Folded Reload
	add $m11, $m11, 80
	br $m10
.Lfunc_end1:
	.size	_ZN13AlievPanfilov7computeEv, .Lfunc_end1-_ZN13AlievPanfilov7computeEv
	.section	.stack_sizes,"Go",@progbits,_ZN13AlievPanfilov7computeEv,comdat,.text._ZN13AlievPanfilov7computeEv
	.long	.Lfunc_begin1
	.byte	80
	.section	.text._ZN13AlievPanfilov7computeEv,"axG",@progbits,_ZN13AlievPanfilov7computeEv,comdat
                                        # -- End function
	.ident	"clang version 13.0.0 (ssh://git@phabricator.sourcevertex.net/diffusion/LLVMPROJECT/llvm-project.git a39a2acfae1c9741d5aa0363a690339a63142bea)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
