//go:build !noasm && arm64

// AUTO-GENERATED BY GOAT -- DO NOT EDIT

package ml

import "unsafe"

//go:noescape
func vmul_const_add_to(a, b, c, n unsafe.Pointer)

//go:noescape
func vmul_const_to(a, b, c, n unsafe.Pointer)

//go:noescape
func vmul_const(a, b, n unsafe.Pointer)

//go:noescape
func vmul_to(a, b, c, n unsafe.Pointer)

//go:noescape
//func vdot(a, b, n, ret unsafe.Pointer)
func vdot(a, b unsafe.Pointer, n uint64, ret unsafe.Pointer)
