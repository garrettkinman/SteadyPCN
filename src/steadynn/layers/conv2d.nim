# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sugar
import ../tensors

type
    Conv2D*[T] = object
        filters*: Tensor[T]
        bias*: Tensor[T]
        stride*: int
        padding*: int
        activation*: (Tensor[T]) -> Tensor[T]

proc newConv2D*[T](filterShape: openArray[int], numFilters: int, stride: int, padding: int, slice: HSlice[T, T], activation: (Tensor[T]) -> Tensor[T]): Conv2D[T] =
    result = Conv2D[T](
        filters: Tensor[T].rand([numFilters] + filterShape, slice),
        bias: Tensor[T].rand([numFilters, 1], slice),
        stride: stride,
        padding: padding,
        activation: activation
    )

proc newConv2D*[T](filters: Tensor[T], bias: Tensor[T], stride: int, padding: int, activation: (Tensor[T]) -> Tensor[T]): Conv2D[T] =
    result = Conv2D[T](
        filters: filters,
        bias: bias,
        stride: stride,
        padding: padding,
        activation: activation
    )

proc newConv2D*[T](filterShape: openArray[int], numFilters: int, stride: int, padding: int, filters: seq[T], bias: seq[T], activation: (Tensor[T]) -> Tensor[T]): Conv2D[T] =
    result = Conv2D[T](
        filters: Tensor[T].new([numFilters] + filterShape, filters),
        bias: Tensor[T].new([numFilters, 1], bias),
        stride: stride,
        padding: padding,
        activation: activation
    )

proc new*[T](_: typedesc[Conv2D[T]], filterShape: openArray[int], numFilters: int, stride: int, padding: int, slice: HSlice[T, T], activation: (Tensor[T]) -> Tensor[T]): Conv2D[T] = newConv2D[T](filterShape, numFilters, stride, padding, slice, activation)
proc new*[T](_: typedesc[Conv2D[T]], filters: Tensor[T], bias: Tensor[T], stride: int, padding: int, activation: (Tensor[T]) -> Tensor[T]): Conv2D[T] = newConv2D(filters, bias, stride, padding, activation)
proc new*[T](_: typedesc[Conv2D[T]], filterShape: openArray[int], numFilters: int, stride: int, padding: int, filters: seq[T], bias: seq[T], activation: (Tensor[T]) -> Tensor[T]): Conv2D[T] = newConv2D(filterShape, numFilters, stride, padding, filters, bias, activation)

# TODO: test
proc forward*[T](self: Conv2D, x: Tensor[T]): Tensor[T] =
    let
        batchSize = x.shape[0]
        inputHeight = x.shape[1]
        inputWidth = x.shape[2]
        inputChannels = x.shape[3]

        filterHeight = self.filters.shape[1]
        filterWidth = self.filters.shape[2]
        numFilters = self.filters.shape[0]

        outputHeight = (inputHeight + 2 * self.padding - filterHeight) div self.stride + 1
        outputWidth = (inputWidth + 2 * self.padding - filterWidth) div self.stride + 1

        output = Tensor[T].zeros([batchSize, outputHeight, outputWidth, numFilters])

    for b in 0..<batchSize:
        for i in 0..<outputHeight:
            for j in 0..<outputWidth:
                for f in 0..<numFilters:
                    var sum: T = 0 # TODO: verify that this is right
                    for c in 0..<inputChannels:
                        for ii in 0..<filterHeight:
                            for jj in 0..<filterWidth:
                                let
                                    xi = i * self.stride + ii - self.padding
                                    xj = j * self.stride + jj - self.padding
                                if xi in 0..<inputHeight and xj in 0..<inputWidth:
                                    sum += x[b, xi, xj, c] * self.filters[f, ii, jj, c]
                    output[b, i, j, f] = sum + self.bias[f, 0]

    result = self.activation(output)