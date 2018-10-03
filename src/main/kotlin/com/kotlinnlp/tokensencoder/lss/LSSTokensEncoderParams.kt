/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.lss

import com.kotlinnlp.lssencoder.LSSParameters
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The parameters of the [LSSTokensEncoderParams].
 *
 * @property lssParams the parameters of the LSS encoder
 */
class LSSTokensEncoderParams(val lssParams: LSSParameters) : TokensEncoderParameters
