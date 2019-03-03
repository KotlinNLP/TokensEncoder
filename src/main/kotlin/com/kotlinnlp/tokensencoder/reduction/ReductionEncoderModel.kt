/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.reduction

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * The model of the [ReductionEncoder].
 *
 * @property inputEncoderModel the model of the input tokens encoder
 * @property tokenEncodingSize the size of the reduced token encoding vectors after
 * @param optimizeInput whether the input encoder has to be optimized (default = true)
 * @param activationFunction the activation function of the reduction network
 * @param weightsInitializer the initializer of the weights of the reduction network (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases of the reduction network (zeros if null, default: Glorot)
 */
class ReductionEncoderModel<TokenType: Token, SentenceType: Sentence<TokenType>>(
  val inputEncoderModel: TokensEncoderModel<TokenType, SentenceType>,
  override val tokenEncodingSize: Int,
  internal val optimizeInput: Boolean = true,
  activationFunction: ActivationFunction?,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : TokensEncoderModel<TokenType, SentenceType> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The network used to reduce the dimension of the tokens encoding vectors.
   */
  val reductionNetwork = StackedLayersParameters(
    LayerInterface(
      size = this.inputEncoderModel.tokenEncodingSize,
      type = LayerType.Input.Dense),
    LayerInterface(
      size = this.tokenEncodingSize,
      connectionType = LayerType.Connection.Feedforward,
      activationFunction = activationFunction),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = "encoding size $tokenEncodingSize"

  /**
   * @param useDropout whether to apply the dropout
   * @param id an identification number useful to track a specific encoder
   *
   * @return a new tokens encoder that uses this model
   */
  override fun buildEncoder(useDropout: Boolean, id: Int) = ReductionEncoder(
    model = this,
    useDropout = useDropout,
    id = id
  )

  /**
   * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
   *
   * @return a new optimizer for this model
   */
  override fun buildOptimizer(updateMethod: UpdateMethod<*>) = ReductionEncoderOptimizer(
    model = this,
    updateMethod = updateMethod
  )
}
