/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * @property parameters the param of the feed-forward network of the [MorphoEncoderModel]
 */
class MorphoEncoderParams(val parameters: NetworkParameters) : TokensEncoderParameters