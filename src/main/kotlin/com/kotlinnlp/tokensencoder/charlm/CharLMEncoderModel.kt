/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charlm

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.*
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * The model of the [CharLMEncoder].
 *
 * @param charLM a char language model trained left to right
 * @param revCharLM a char language model trained right to left
 * @param outputMergeConfiguration the configuration of the output merge layer
 * @param weightsInitializer the initializer of the weights of the merge layer (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases of the merge layer (zeros if null, default: null)
 */
class CharLMEncoderModel(
  val charLM: CharLM,
  val revCharLM: CharLM,
  outputMergeConfiguration: MergeConfiguration = ConcatMerge(),
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : TokensEncoderModel<FormToken, Sentence<FormToken>> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  init {
    require(!this.charLM.reverseModel) { "The charLM must be trained to process the sequence from left to right."}
    require(this.revCharLM.reverseModel) { "The revCharLM must be trained to process the sequence from right to left."}
    require(this.charLM.recurrentNetwork.outputSize == this.revCharLM.recurrentNetwork.outputSize) {
      "The charLM and the reverse CharLM must have the same recurrent hidden size."
    }
  }

  /**
   * The size of the token encoding vectors.
   */
  override val tokenEncodingSize: Int = when (outputMergeConfiguration) {
    is AffineMerge -> outputMergeConfiguration.outputSize
    is BiaffineMerge -> outputMergeConfiguration.outputSize
    is ConcatFeedforwardMerge -> outputMergeConfiguration.outputSize
    is ConcatMerge -> 2 * this.charLM.recurrentNetwork.outputSize
    is SumMerge -> this.charLM.recurrentNetwork.outputSize
    is ProductMerge -> this.charLM.recurrentNetwork.outputSize
    is AvgMerge -> this.charLM.recurrentNetwork.outputSize
    else -> throw RuntimeException("Invalid output merge configuration.")
  }

  /**
   * The Merge network that combines the predictions of the two language models.
   */
  val outputMergeNetwork = StackedLayersParameters(
    if (outputMergeConfiguration is ConcatFeedforwardMerge) listOf(
      LayerInterface(
        sizes = listOf(this.charLM.recurrentNetwork.outputSize, this.revCharLM.recurrentNetwork.outputSize),
        dropout = outputMergeConfiguration.dropout),
      LayerInterface(size = 2 * this.charLM.recurrentNetwork.outputSize, connectionType = LayerType.Connection.Concat),
      LayerInterface(
        size = outputMergeConfiguration.outputSize,
        activationFunction = outputMergeConfiguration.activationFunction,
        connectionType = LayerType.Connection.Feedforward))
    else listOf(
      LayerInterface(
        sizes = listOf(this.charLM.recurrentNetwork.outputSize, this.revCharLM.recurrentNetwork.outputSize),
        dropout = outputMergeConfiguration.dropout),
      LayerInterface(size = this.tokenEncodingSize, connectionType = outputMergeConfiguration.type)),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * @param useDropout whether to apply the dropout
   * @param id an identification number useful to track a specific encoder
   *
   * @return a new tokens encoder that uses this model
   */
  override fun buildEncoder(useDropout: Boolean, id: Int) = CharLMEncoder(model = this, id = id)
}
