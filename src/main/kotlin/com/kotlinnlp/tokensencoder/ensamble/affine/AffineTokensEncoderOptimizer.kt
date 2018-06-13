/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.affine

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.layers.models.merge.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizerFactory
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The optimizer of the [AffineTokensEncoder]
 *
 * @param model the model to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...
 */
open class AffineTokensEncoderOptimizer(
  private val model: AffineTokensEncoderModel,
  updateMethod: UpdateMethod<*>
) : TokensEncoderOptimizer(
  model = model,
  updateMethod = updateMethod
) {

  /**
   * The optimizers of the encoders.
   */
  private val encodersOptimizers = this.model.models.map { TokensEncoderOptimizerFactory(it, updateMethod) }

  /**
   * The optimizer of the affine layer.
   */
  private val affineLayerOptimizer: ParamsOptimizer<AffineLayerParameters> =
    ParamsOptimizer(this.model.affineParams, updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {

    this.affineLayerOptimizer.update()
    this.encodersOptimizers.forEach { it.update() }
  }

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: TokensEncoderParameters, copy: Boolean) {

    paramsErrors as AffineTokensEncoderParams

    this.affineLayerOptimizer.accumulate(paramsErrors.affineParams)

    paramsErrors.encodersParams.forEachIndexed { index, values ->
      this.encodersOptimizers[index].accumulate(values, copy)
    }
  }
}