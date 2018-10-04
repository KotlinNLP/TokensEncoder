/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.lss

import com.kotlinnlp.lssencoder.LSSOptimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The optimizer of the [LSSTokensEncoder].
 *
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 * @param model the model of this optimizer
 */
class LSSTokensEncoderOptimizer(
  private val model: LSSTokensEncoderModel<*, *>,
  updateMethod: UpdateMethod<*>
) : TokensEncoderOptimizer(
  model = model,
  updateMethod = updateMethod
) {

  /**
   * The optimizer of the encoder parameters.
   */
  private val lssOptimizer = LSSOptimizer(model = this.model.lssModel, updateMethod = this.updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {
    this.lssOptimizer.update()
  }

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: TokensEncoderParameters, copy: Boolean) {
    this.lssOptimizer.accumulate(paramsErrors = (paramsErrors as LSSTokensEncoderParams).lssParams, copy = copy)
  }
}
