/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.concat

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizerFactory
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The optimizer of the [ConcatTokensEncoder]
 *
 * @param model the model to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...
 */
open class ConcatTokensEncoderOptimizer(
  private val model: ConcatTokensEncoderModel,
  updateMethod: UpdateMethod<*>
) : TokensEncoderOptimizer(
  model = model,
  updateMethod = updateMethod
) {

  /**
   * The optimizer of the word embeddings map.
   */
  private val optimizers = this.model.models.map { TokensEncoderOptimizerFactory(it, updateMethod) }

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() = this.optimizers.forEach { it.update() }

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: TokensEncoderParameters, copy: Boolean) {

    paramsErrors as ConcatTokensEncoderParams

    paramsErrors.params.forEachIndexed { index, values ->
      this.optimizers[index].accumulate(values, copy)
    }
  }
}