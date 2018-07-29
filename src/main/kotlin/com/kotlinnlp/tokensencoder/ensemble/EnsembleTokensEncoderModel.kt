/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensemble

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.*
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * The model of the [EnsembleTokensEncoderModel].
 *
 * @property models the list of tokens-encoder models
 * @param outputMergeConfiguration the configuration of the output merge layer
 * @param weightsInitializer the initializer of the output merge network weights
 * @param biasesInitializer the initializer of the output merge network biases
 */
open class EnsembleTokensEncoderModel(
  val models: List<TokensEncoderModel<*, *>>,
  outputMergeConfiguration: MergeConfiguration = ConcatMerge(),
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : TokensEncoderModel<Token, Sentence<Token>> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  private val mergeOutputSize: Int = when (outputMergeConfiguration) {
    is AffineMerge -> outputMergeConfiguration.outputSize
    is BiaffineMerge -> outputMergeConfiguration.outputSize
    is ConcatFeedforwardMerge -> outputMergeConfiguration.outputSize
    is ConcatMerge -> this.models.sumBy { it.tokenEncodingSize }
    is SumMerge, is ProductMerge, is AvgMerge -> {
      require(this.models.all { it.tokenEncodingSize == this.models[0].tokenEncodingSize } )
      this.models[0].tokenEncodingSize
    }
    else -> throw RuntimeException("Invalid output merge configuration.")
  }

  /**
   * The size of the token encoding vectors.
   */
  override val tokenEncodingSize: Int = this.mergeOutputSize

  /**
   * The Merge network that combines encoded vectors of each encoder.
   */
  val outputMergeNetwork = NeuralNetwork(
    if (outputMergeConfiguration is ConcatFeedforwardMerge) listOf(
      LayerInterface(sizes = this.models.map { it.tokenEncodingSize }, dropout = outputMergeConfiguration.dropout),
      LayerInterface(size = this.models.sumBy { it.tokenEncodingSize }, connectionType = LayerType.Connection.Concat),
      LayerInterface(size = outputMergeConfiguration.outputSize, connectionType = LayerType.Connection.Feedforward))
    else listOf(
      LayerInterface(sizes = this.models.map { it.tokenEncodingSize }, dropout = outputMergeConfiguration.dropout),
      LayerInterface(size = this.mergeOutputSize, connectionType = outputMergeConfiguration.type)),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = """
    encoding size %d
  """.trimIndent().format(
    this.tokenEncodingSize
  )
}
