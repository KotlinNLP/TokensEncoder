/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.feedforward

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderParameters
import com.kotlinnlp.tokensencoder.ensamble.concat.ConcatTokensEncoderOptimizer

/**
 * The optimizer of the [FFTokensEncoder].
 *
 * @param model the model of this optimizer
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class FFTokensEncoderOptimizer(
  private val model: FFTokensEncoderModel,
  updateMethod: UpdateMethod<*>
) : ConcatTokensEncoderOptimizer(
  model = model,
  updateMethod = updateMethod
) {

  /**
   * The optimizer of the feed-forward network used to reduce the concatenate vectors.
   */
  private val networkOptimizer: ParamsOptimizer<NetworkParameters> =
    ParamsOptimizer(this.model.tokenEncodingNetwork.network.model, updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {

    super.update()
    this.networkOptimizer.update()
  }

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: TokensEncoderParameters, copy: Boolean) {

    paramsErrors as FFTokensEncoderParams

    super.accumulate(paramsErrors, copy)
    this.networkOptimizer.accumulate(paramsErrors.networkParams, copy)
  }
}