/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charactersattention

import com.kotlinnlp.simplednn.deeplearning.attention.han.HANParameters
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The parameters of the [CharsAttentionEncoder].
 *
 * @property hanParameters list of [HANParameters]
 * @property embeddingsParams list of [Embedding]s
 */
class CharsAttentionEncoderParams(
  val hanParameters: List<HANParameters>,
  val embeddingsParams: List<Embedding>
) : TokensEncoderParameters