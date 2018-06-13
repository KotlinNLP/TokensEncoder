/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.affine

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.merge.affine.AffineLayerParameters
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import java.io.Serializable

/**
 * The model of the [AffineTokensEncoder].
 *
 * @property models the list of tokens-encoder models
 * @property tokenEncodingSize the size of the token encoding vectors.
 * @property activation the activation function of the affine layer (can be null)
 */
class AffineTokensEncoderModel(
  val models: List<TokensEncoderModel>,
  override val tokenEncodingSize: Int,
  internal val activation: ActivationFunction?,
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
   * The parameters of the affine layer.
   */
  val affineParams = AffineLayerParameters(
    inputsSize = this.models.map { it.tokenEncodingSize },
    outputSize = this.tokenEncodingSize,
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
