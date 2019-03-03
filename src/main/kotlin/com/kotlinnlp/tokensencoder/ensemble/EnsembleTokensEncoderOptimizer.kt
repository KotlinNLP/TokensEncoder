/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensemble

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The optimizer of the [EnsembleTokensEncoderOptimizer]
 *
 * @param model the model to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
open class EnsembleTokensEncoderOptimizer(
  private val model: EnsembleTokensEncoderModel<*, *>,
  updateMethod: UpdateMethod<*>
) : TokensEncoderOptimizer(
  model = model,
  updateMethod = updateMethod
) {

  /**
   * The list of optimizers of the ensemble encoders models.
   */
  private val encodersOptimizers: List<TokensEncoderOptimizer?> = this.model.components.map {
    if (it.trainable) it.buildOptimizer(updateMethod) else null
  }

  /**
   * The optimizer of the output merge network.
   */
  private val outputMergeOptimizer = ParamsOptimizer(this.model.outputMergeNetwork, updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {
    this.outputMergeOptimizer.update()
    this.encodersOptimizers.forEach { it?.update() }
  }

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: TokensEncoderParameters, copy: Boolean) {

    paramsErrors as EnsembleTokensEncoderParams

    this.outputMergeOptimizer.accumulate(paramsErrors.outputMergeParams, copy = copy)

    paramsErrors.encodersParams.forEachIndexed { index, values ->
      this.encodersOptimizers[index]?.accumulate(values, copy)
    }
  }
}
