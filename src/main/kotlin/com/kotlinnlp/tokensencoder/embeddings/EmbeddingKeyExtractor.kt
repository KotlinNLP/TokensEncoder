/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import java.io.Serializable

/**
 * Extracts the string to use as embedding key from the token of a sentence.
 */
interface EmbeddingKeyExtractor<TokenType: Token, SentenceType: Sentence<TokenType>> : Serializable {

  /**
   * @param sentence a generic sentence
   * @param tokenId the id of the token from which to extract the key
   *
   * @return the string to use as embedding key
   */
  fun getKey(sentence: SentenceType, tokenId: Int): String
}
