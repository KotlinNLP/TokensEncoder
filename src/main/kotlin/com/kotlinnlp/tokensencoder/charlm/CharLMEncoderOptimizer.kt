/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charlm

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The optimizer of the [CharLMEncoder].
 *
 * @param model the model to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class CharLMEncoderOptimizer(
  private val model: CharLMEncoderModel,
  updateMethod: UpdateMethod<*>
) : TokensEncoderOptimizer(
  model = model,
  updateMethod = updateMethod
) {

  /**
   * Update the parameters of the neural element associated to this optimizer.
   *
   * It has no effect in this implementation.
   */
  override fun update() = Unit

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * It has no effect in this implementation.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: TokensEncoderParameters, copy: Boolean) = Unit
}