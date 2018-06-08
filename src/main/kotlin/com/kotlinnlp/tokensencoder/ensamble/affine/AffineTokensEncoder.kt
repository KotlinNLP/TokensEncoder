/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.affine

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.core.layers.merge.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.core.layers.merge.affine.AffineLayerStructure
import com.kotlinnlp.simplednn.core.layers.merge.affine.AffineLayersPool
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderBuilder
import com.kotlinnlp.tokensencoder.TokensEncoderFactory
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The tokens-encoder that encodes a token performing an affine transformation of the results of other [TokensEncoder]s.
 *
 * @property model the model of this tokens encoder
 * @property trainingMode whether the encoder is being trained
 */
open class AffineTokensEncoder(
  private val model: AffineTokensEncoderModel,
  private val trainingMode: Boolean
) : TokensEncoder {

  /**
   * List of tokens encoder builders.
   */
  private val encodersBuilder: List<TokensEncoderBuilder> = model.models.map {
    require(it !is AffineTokensEncoderModel) // avoid recursion
    TokensEncoderFactory(it, this.trainingMode)
  }

  /**
   * The list of [TokensEncoder] used in the last encoding.
   */
  private val usedEncoders = mutableListOf<TokensEncoder>()

  /**
   * A pool of Affine Layers used to build the attention arrays.
   */
  private val affineLayersPool = AffineLayersPool<DenseNDArray>(
    params = this.model.affineParams,
    activationFunction = this.model.activation)

  /**
   * The affine layers used in the last encoding.
   */
  private val usedAffineLayers = mutableListOf<AffineLayerStructure<DenseNDArray>>()

  /**
   * The structure used to store the params errors of the affine layers during the backward.
   */
  private lateinit var affineLayerParamsErrors: AffineLayerParameters

  /**
   * The params errors accumulator of the affine layers.
   */
  private var affineErrorsAccumulator = ParamsErrorsAccumulator<AffineLayerParameters>()

  /**
   * Encode a list of tokens.
   *
   * @param tokens a list of [Token]
   *
   * @return a list of the same size of the [tokens] with their encoded representation
   */
  override fun encode(tokens: List<Token>): List<DenseNDArray> {

    this.reset() // reset forward and backward variables

    this.usedEncoders.addAll(this.encodersBuilder.map { it.invoke() })

    val tokenEncodings = List<MutableList<DenseNDArray>>(size = tokens.size, init = { mutableListOf() })

    this.usedEncoders.forEach { encoder ->

      encoder.encode(tokens).forEachIndexed { tokenId, values ->
        tokenEncodings[tokenId].add(values)
      }
    }

    return tokenEncodings.map { this.doAffineTransform(it) }
  }

  /**
   * Propagate the errors.
   *
   * @param errors the errors of the current encoding
   */
  override fun backward(errors: List<DenseNDArray>) {

    val partErrors: List<List<DenseNDArray>> = errors.mapIndexed { index, values ->
      this.backwardAffineLayer(this.usedAffineLayers[index], outputErrors = values)
    }

    this.affineErrorsAccumulator.averageErrors()

    this.usedEncoders.forEachIndexed { encoderIndex, encoder ->
      encoder.backward(partErrors.map { it[encoderIndex] } )
    }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [TokensEncoderParameters] parameters
   */
  override fun getParamsErrors(copy: Boolean): TokensEncoderParameters =
    AffineTokensEncoderParams(
      encodersParams = this.usedEncoders.map { it.getParamsErrors(copy = copy) },
      affineParams = this.affineErrorsAccumulator.getParamsErrors(copy = copy))

  /**
   * @return the result of the affine transformation
   */
  private fun doAffineTransform(inputVectors: List<DenseNDArray>): DenseNDArray {

    val affineLayer = this.getAffineLayer()

    inputVectors.forEachIndexed { index, values ->
      affineLayer.setInput(index, values)
    }

    affineLayer.forward()

    return affineLayer.outputArray.values
  }

  /**
   * @return an affine layer
   */
  private fun getAffineLayer(): AffineLayerStructure<DenseNDArray> {

    this.usedAffineLayers.add(this.affineLayersPool.getItem())
    return this.usedAffineLayers.last()
  }

  /**
   * A single affine layer backward.
   *
   * @param layer a transform layer
   * @param outputErrors the errors of the output
   *
   * @return the errors of the input
   */
  private fun backwardAffineLayer(layer: AffineLayerStructure<DenseNDArray>,
                                  outputErrors: DenseNDArray): List<DenseNDArray> {

    val paramsErrors = this.getAffineParamsErrors()

    layer.setErrors(outputErrors)
    layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

    this.affineErrorsAccumulator.accumulate(paramsErrors)

    return layer.getInputErrors(copy = true)
  }

  /**
   * @return the affine layers params errors
   */
  private fun getAffineParamsErrors(): AffineLayerParameters {

    if (!this::affineLayerParamsErrors.isInitialized) {
      this.affineLayerParamsErrors = this.usedAffineLayers.last().params.copy()
    }

    return this.affineLayerParamsErrors
  }

  /**
   * Reset forward and backward variables.
   */
  private fun reset() {
    this.affineLayersPool.releaseAll()
    this.usedAffineLayers.clear()
    this.usedEncoders.clear()
    this.affineErrorsAccumulator.reset()
  }
}
