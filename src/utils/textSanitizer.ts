/**
 * Polishes and sanitizes Portuguese captions, instructions, and voiceover texts.
 * Fixes common LLM repetition and syntax glitches:
 * - "na para tigela" -> "na tigela para começar"
 * - "na para" -> "na"
 * - "para na" -> "para a"
 * - "até e inflar dourar" -> "até inflar e dourar"
 * - "obter e uma" -> "obter uma"
 * - "Pão de queijo dos deuses! e Salva me segue agora!" -> "Pão de queijo dos deuses! Salve e me siga agora!"
 * - Double words ("de de", "o o", "a a")
 */
export function cleanAndPolishPortugueseCaption(raw: string): string {
  if (!raw || typeof raw !== 'string') return '';
  let s = raw.trim();

  // 1. Fix "na para tigela começar" -> "na tigela para começar"
  s = s.replace(/\bna\s+para\s+(\w+)\s+(\w+)\b/gi, 'na $1 para $2');
  s = s.replace(/\bno\s+para\s+(\w+)\s+(\w+)\b/gi, 'no $1 para $2');
  s = s.replace(/\bna\s+para\b/gi, 'na');
  s = s.replace(/\bpara\s+na\b/gi, 'para a');
  s = s.replace(/\bno\s+para\b/gi, 'no');
  s = s.replace(/\bpara\s+no\b/gi, 'para o');
  s = s.replace(/\bde\s+para\b/gi, 'de');
  s = s.replace(/\bem\s+na\b/gi, 'na');
  s = s.replace(/\bem\s+no\b/gi, 'no');

  // 2. Fix misplaced conjunctions: "até e inflar dourar" -> "até inflar e dourar"
  s = s.replace(/\baté\s+e\s+(\w+)\s+(\w+)/gi, 'até $1 e $2');
  s = s.replace(/\baté\s+e\s+/gi, 'até ');
  s = s.replace(/\bobter\s+e\s+uma\b/gi, 'obter uma');
  s = s.replace(/\bobter\s+e\s+um\b/gi, 'obter um');
  s = s.replace(/\bficar\s+e\s+(\w+)/gi, 'ficar $1');

  // 3. Fix CTA connectors: "! e Salva me segue agora!" -> "! Salve e me siga agora!"
  s = s.replace(/[!.]\s*e\s+Salva\s+me\s+segue\s+agora!?/gi, '! Salve e me siga agora!');
  s = s.replace(/[!.]\s*e\s+salva\s+e\s+segue!?/gi, '! Salve e me siga!');
  s = s.replace(/\bSalva\s+me\s+segue\s+agora!?/gi, 'Salve e me siga agora!');
  s = s.replace(/\bSalva\s+me\s+segue\b/gi, 'Salve e me siga');
  s = s.replace(/\bcurte\s+e\s+salva\s+me\s+segue\b/gi, 'Curta, salve e me siga');

  // 4. Fix duplicated consecutive words (e.g., "queijo queijo", "a a", "o o")
  s = s.replace(/\b([a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]{2,})\s+\1\b/gi, '$1');

  // 5. Clean up multiple spaces and weird punctuation
  s = s.replace(/\s{2,}/g, ' ');
  s = s.replace(/\s+([,!?.])/g, '$1');

  // 6. Ensure first letter is capitalized
  if (s.length > 0) {
    s = s.charAt(0).toUpperCase() + s.slice(1);
  }

  return s;
}
