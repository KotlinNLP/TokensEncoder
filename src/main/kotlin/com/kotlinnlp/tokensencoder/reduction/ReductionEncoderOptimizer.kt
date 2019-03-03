/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.reduction

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The optimizer of the [ReductionEncoder].
 *
 * @param model the model to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class ReductionEncoderOptimizer(
  private val model: ReductionEncoderModel<*, *>,
  updateMethod: UpdateMethod<*>
) : TokensEncoderOptimizer(
  model = model,
  updateMethod = updateMethod
) {

  /**
   * The optimizer of the parameters of the input tokens encoder (null if the input encoder must not be trained).
   */
  private val inputOptimizer: TokensEncoderOptimizer? =
    if (this.model.optimizeInput) this.model.inputEncoderModel.buildOptimizer(updateMethod) else null

  /**
   * The optimizer of the parameters of the reduction network.
   */
  private val reductionOptimizer: ParamsOptimizer<StackedLayersParameters> =
    ParamsOptimizer(params = this.model.reductionNetwork, updateMethod = updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {

    this.inputOptimizer?.update()
    this.reductionOptimizer.update()
  }

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: TokensEncoderParameters, copy: Boolean) {

    paramsErrors as ReductionEncoderParams

    this.inputOptimizer?.accumulate(paramsErrors.inputParams!!)
    this.reductionOptimizer.accumulate(paramsErrors.reductionParams)
  }
}
