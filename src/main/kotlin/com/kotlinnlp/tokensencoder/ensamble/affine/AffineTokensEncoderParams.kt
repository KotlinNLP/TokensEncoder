/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.affine

import com.kotlinnlp.simplednn.core.mergelayers.affine.AffineLayerParameters
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The parameters of the [AffineTokensEncoder].
 *
 * @param encodersParams the parameters of the encoders
 * @param affineParams the parameters of the affine layer
 */
open class AffineTokensEncoderParams(
  val encodersParams: List<TokensEncoderParameters>,
  val affineParams: AffineLayerParameters
) : TokensEncoderParameters