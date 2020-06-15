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
 * @param dirCharLM a char language model trained left to right
 * @param revCharLM a char language model trained right to left
 * @param outputMergeConfiguration the configuration of the output merge layer
 * @param weightsInitializer the initializer of the weights of the merge layer (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases of the merge layer (zeros if null, default: null)
 */
class CharLMEncoderModel(
  val dirCharLM: CharLM,
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
    require(this.dirCharLM.hiddenNetwork.outputSize == this.revCharLM.hiddenNetwork.outputSize) {
      "The direct and the reverse char language models must have the same recurrent hidden size."
    }
  }

  /**
   * The size of the token encoding vectors.
   */
  override val tokenEncodingSize: Int = when (outputMergeConfiguration) {
    is AffineMerge -> outputMergeConfiguration.outputSize
    is BiaffineMerge -> outputMergeConfiguration.outputSize
    is ConcatFeedforwardMerge -> outputMergeConfiguration.outputSize
    is ConcatMerge -> 2 * this.dirCharLM.hiddenNetwork.outputSize
    is SumMerge -> this.dirCharLM.hiddenNetwork.outputSize
    is ProductMerge -> this.dirCharLM.hiddenNetwork.outputSize
    is AvgMerge -> this.dirCharLM.hiddenNetwork.outputSize
    else -> throw RuntimeException("Invalid output merge configuration.")
  }

  /**
   * The Merge network that combines the predictions of the two language models.
   */
  val outputMergeNetwork = StackedLayersParameters(
    layersConfiguration = if (outputMergeConfiguration is ConcatFeedforwardMerge)
      listOf(
        LayerInterface(
          sizes = listOf(this.dirCharLM.hiddenNetwork.outputSize, this.revCharLM.hiddenNetwork.outputSize)),
        LayerInterface(
          size = 2 * this.dirCharLM.hiddenNetwork.outputSize,
          connectionType = LayerType.Connection.Concat),
        LayerInterface(
          size = outputMergeConfiguration.outputSize,
          activationFunction = outputMergeConfiguration.activationFunction,
          connectionType = LayerType.Connection.Feedforward))
    else
      listOf(
        LayerInterface(
          sizes = listOf(this.dirCharLM.hiddenNetwork.outputSize, this.revCharLM.hiddenNetwork.outputSize)),
        LayerInterface(
          size = this.tokenEncodingSize,
          connectionType = outputMergeConfiguration.type)),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * @param id an identification number useful to track a specific encoder
   *
   * @return a new tokens encoder that uses this model
   */
  override fun buildEncoder(id: Int) = CharLMEncoder(model = this, id = id)
}
