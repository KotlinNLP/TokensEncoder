/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.feedforward

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.tokensencoder.ensamble.concat.ConcatTokensEncoderModel

/**
 * The model of the [FFTokensEncoder].
 *
 * @property models the list of tokens-encoder models
 * @property tokenEncodingSize the size of the token encoding vectors
 * @property activation the activation function of the feed-forward network (can be null)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
 */
class FFTokensEncoderModel(
  models: List<TokensEncoderModel>,
  override val tokenEncodingSize: Int,
  private val activation: ActivationFunction?,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : ConcatTokensEncoderModel(models = models) {

  /**
   * The network for the output tokens encodings.
   */
  val tokenEncodingNetwork = NeuralNetwork (
    LayerConfiguration(
      size = this.models.sumBy { it.tokenEncodingSize },
      inputType = LayerType.Input.Dense),
    LayerConfiguration(
      size = this.tokenEncodingSize,
      activationFunction = this.activation,
      connectionType = LayerType.Connection.Feedforward
    ),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )
}