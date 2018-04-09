/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.concat

import com.kotlinnlp.tokensencoder.TokensEncoderModel
import java.io.Serializable

/**
 * The model of the [ConcatTokensEncoder].
 *
 * @property models the list of tokens-encoder models
 */
open class ConcatTokensEncoderModel(
  val models: List<TokensEncoderModel>
) : TokensEncoderModel, Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The size of the token encoding vectors.
   */
  override val tokenEncodingSize: Int = this.models.sumBy { it.tokenEncodingSize }

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = """
    encoding size %d
  """.trimIndent().format(
    this.tokenEncodingSize
  )
}
