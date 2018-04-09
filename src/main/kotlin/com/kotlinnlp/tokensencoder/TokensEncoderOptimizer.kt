/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.Optimizer

/**
 * The tokens-encoder optimizer.
 *
 * @param model the model to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
abstract class TokensEncoderOptimizer(
  private val model: TokensEncoderModel,
  updateMethod: UpdateMethod<*>
) : Optimizer(
  updateMethod = updateMethod
) {

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  abstract fun accumulate(paramsErrors: TokensEncoderParameters, copy: Boolean = true)
}