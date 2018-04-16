/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho

import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.deeplearning.sequenceencoder.SequenceFeedforwardEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderParameters
import com.kotlinnlp.neuraltokenizer.Token as TKToken

/**
 * The [TokensEncoder] that encodes each token of a sentence using its morphological properties.
 *
 * @property model the model of this tokens encoder
 * @property trainingMode whether the encoder is being trained
 */
class MorphoEncoder(
  private val model: MorphoEncoderModel,
  private val trainingMode: Boolean
) : TokensEncoder {

  /**
   * The feed-forward network used to transform the input from sparse to dense.
   */
  private val encoder = SequenceFeedforwardEncoder<SparseBinaryNDArray>(this.model.denseEncoder)

  /**
   * Encode a list of tokens.
   *
   * @param tokens a list of [Token]
   *
   * @return a list of the same size of the [tokens] with their encoded representation
   */
  override fun encode(tokens: List<Token>): Array<DenseNDArray> {

    val tokenFeatures = FeaturesExtractor(
      tokens = tokens,
      analyzer = MorphologicalAnalyzer(this.model.dictionary),
      langCode = this.model.langCode).extractFeatures()

    return this.encoder.encode(Array(size = tokens.size, init = {

      SparseBinaryNDArrayFactory.arrayOf(
        activeIndices = tokenFeatures[it].getActiveFeaturesIndicies(),
        shape = Shape(this.model.featuresDictionary.size))
    }))
  }

  /**
   * Propagate the errors.
   *
   * @param errors the errors of the current encoding
   */
  override fun backward(errors: Array<DenseNDArray>) = this.encoder.backward(errors, propagateToInput = false)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [TokensEncoderParameters] parameters
   */
  override fun getParamsErrors(copy: Boolean): TokensEncoderParameters =
    MorphoEncoderParams(feedforwardParameters = this.encoder.getParamsErrors(copy = copy))

  /**
   * Map the features set to the features ids using the featuresDictionary.
   *
   * @return the indexes of the active features
   */
  private fun Set<String>.getActiveFeaturesIndicies(): IntArray {

    val activeIndicies = mutableListOf<Int>()

    this.forEach { this@MorphoEncoder.model.featuresDictionary.getId(it)?.let { activeIndicies.add(it) } }

    return activeIndicies.toIntArray()
  }
}