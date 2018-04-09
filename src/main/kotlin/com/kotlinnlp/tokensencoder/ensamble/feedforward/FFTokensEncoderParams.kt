/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.feedforward

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.tokensencoder.TokensEncoderParameters
import com.kotlinnlp.tokensencoder.ensamble.concat.ConcatTokensEncoderParams

/**
 * The parameters of the [FFTokensEncoder].
 *
 * @param encodersParams list of tokens-encoder parameters
 * @property networkParams the feed-forward output network parameters
 */
class FFTokensEncoderParams(
  encodersParams: List<TokensEncoderParameters>,
  val networkParams: NetworkParameters
) : ConcatTokensEncoderParams(
  params = encodersParams
)