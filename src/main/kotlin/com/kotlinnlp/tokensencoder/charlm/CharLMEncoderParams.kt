/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charlm

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The parameters of the [CharLMEncoder].
 *
 * @property mergeNetworkParameters the parameters of the output merge layer
 */
class CharLMEncoderParams(val mergeNetworkParameters: StackedLayersParameters) : TokensEncoderParameters