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
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.*
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * The model of the [EnsembleTokensEncoderModel].
 *
 * @property components the list of ensemble components (tokens encoder models)
 * @param outputMergeConfiguration the configuration of the output merge layer
 * @param weightsInitializer the initializer of the output merge network weights
 * @param biasesInitializer the initializer of the output merge network biases
 */
class EnsembleTokensEncoderModel<TokenType: Token, SentenceType: Sentence<TokenType>>(
  val components: List<ComponentModel<TokenType, SentenceType>>,
  outputMergeConfiguration: MergeConfiguration = ConcatMerge(),
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : TokensEncoderModel<TokenType, SentenceType> {

  /**
   * The model of a component.
   *
   * @param model the model of a tokens encoder
   * @property trainable whether to train the model (default = true)
   */
  class ComponentModel<TokenType: Token, SentenceType: Sentence<TokenType>>(
    model: TokensEncoderModel<TokenType, SentenceType>,
    val trainable: Boolean = true
  ) : TokensEncoderModel<TokenType, SentenceType> by model

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The output size of the output merge layer.
   */
  private val mergeOutputSize: Int = when (outputMergeConfiguration) {
    is AffineMerge -> outputMergeConfiguration.outputSize
    is BiaffineMerge -> outputMergeConfiguration.outputSize
    is ConcatFeedforwardMerge -> outputMergeConfiguration.outputSize
    is ConcatMerge -> this.components.sumBy { it.tokenEncodingSize }
    is SumMerge, is ProductMerge, is AvgMerge -> {
      require(this.components.all { it.tokenEncodingSize == this.components[0].tokenEncodingSize })
      this.components[0].tokenEncodingSize
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
  val outputMergeNetwork = StackedLayersParameters(
    layersConfiguration = if (outputMergeConfiguration is ConcatFeedforwardMerge)
      listOf(
        LayerInterface(
          sizes = this.components.map { it.tokenEncodingSize },
          dropout = outputMergeConfiguration.dropout),
        LayerInterface(
          size = this.components.sumBy { it.tokenEncodingSize },
          connectionType = LayerType.Connection.Concat),
        LayerInterface(
          size = outputMergeConfiguration.outputSize,
          connectionType = LayerType.Connection.Feedforward))
    else
      listOf(
        LayerInterface(
          sizes = this.components.map { it.tokenEncodingSize },
          dropout = outputMergeConfiguration.dropout),
        LayerInterface(
          size = this.mergeOutputSize,
          connectionType = outputMergeConfiguration.type)),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = "encoding size %d".format(this.tokenEncodingSize)

  /**
   * @param useDropout whether to apply the dropout
   * @param id an identification number useful to track a specific encoder
   *
   * @return a new tokens encoder that uses this model
   */
  override fun buildEncoder(useDropout: Boolean, id: Int) = EnsembleTokensEncoder(
    model = this,
    useDropout = useDropout,
    id = id
  )
}
