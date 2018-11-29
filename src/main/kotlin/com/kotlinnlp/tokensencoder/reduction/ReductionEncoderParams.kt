/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.reduction

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The parameters of the [ReductionEncoder].
 *
 * @property inputParams the parameters of the input tokens encoder
 * @property reductionParams the parameters of the reduction network
 */
class ReductionEncoderParams(
  val inputParams: TokensEncoderParameters,
  val reductionParams: NetworkParameters
) : TokensEncoderParameters
