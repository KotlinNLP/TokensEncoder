/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings

import com.kotlinnlp.neuralparser.language.Token

/**
 * The extractor of a token property to be used as embeddings key.
 */
interface TokenEmbeddingKey {

  /**
   * @param token a token
   *
   * @return the string representation of the key [token] property
   */
  operator fun invoke(token: Token): String
}