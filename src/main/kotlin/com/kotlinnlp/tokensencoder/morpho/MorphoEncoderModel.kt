/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho

import com.kotlinnlp.linguisticdescription.morphology.dictionary.MorphologyDictionary
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.sequenceencoder.SequenceFeedforwardNetwork
import com.kotlinnlp.simplednn.utils.DictionarySet
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import java.io.Serializable

/**
 * @property langCode
 * @property dictionary
 * @property featuresDictionary
 * @property tokenEncodingSize the size of the token encoding vectors.
 */
class MorphoEncoderModel(
  val langCode: String,
  val dictionary: MorphologyDictionary,
  val featuresDictionary: DictionarySet<String>,
  override val tokenEncodingSize: Int,
  activation: ActivationFunction?,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : TokensEncoderModel, Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The model of the feed-forward Network used to transform the input from sparse to dense
   */
  val denseEncoder = SequenceFeedforwardNetwork(
    inputType = LayerType.Input.SparseBinary,
    inputSize = this.featuresDictionary.size,
    outputSize = this.tokenEncodingSize,
    outputActivation = activation,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)
}