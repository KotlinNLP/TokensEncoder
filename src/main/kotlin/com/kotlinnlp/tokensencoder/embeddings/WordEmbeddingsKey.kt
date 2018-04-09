/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings

import com.kotlinnlp.neuralparser.language.Token
import java.io.Serializable

/**
 * The token embedding key extractor that extracts the normalized word as property key.
 */
object WordEmbeddingsKey : TokenEmbeddingKey, Serializable {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * @param token a token
   *
   * @return the string representation of the key [token] property
   */
  override fun invoke(token: Token): String = token.normalizedWord
}