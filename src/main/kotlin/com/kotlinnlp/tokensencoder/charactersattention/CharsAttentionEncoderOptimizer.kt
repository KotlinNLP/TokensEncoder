/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charactersattention

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The optimizer of the [CharsAttentionEncoder].
 *
 * @param model the model to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class CharsAttentionEncoderOptimizer(
  private val model: CharsAttentionEncoderModel,
  updateMethod: UpdateMethod<*>
) : TokensEncoderOptimizer(
  model = model,
  updateMethod = updateMethod
) {

  /**
   * The optimizer of the characters network.
   */
  private val hanOptimizer = ParamsOptimizer(this.model.charactersNetwork.params, this.updateMethod)

  /**
   * The optimizer of the characters embeddings map.
   */
  private val embeddingsOptimizer = EmbeddingsOptimizer(this.model.charsEmbeddings, this.updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {

    this.hanOptimizer.update()
    this.embeddingsOptimizer.update()
  }

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: TokensEncoderParameters, copy: Boolean) {

    paramsErrors as CharsAttentionEncoderParams

    paramsErrors.hanParameters.forEach { this.hanOptimizer.accumulate(it, copy = copy) }
    paramsErrors.embeddingsParams.forEach { this.embeddingsOptimizer.accumulate(it, errors = it.array.values) }
  }
}