# IDRE-Angular — Threat Model & Security Guarantees (v1.0)

**Propósito**: Definir los límites, capacidades y restricciones del sistema para demostrar formalmente por qué IDRE-Angular es seguro, no-trazable y dependiente del campo.

---

## 1. Supuestos del Sistema
Para que el protocolo funcione como está diseñado, asumimos:
*   **Estabilidad**: El attractor del receptor es estable (Demostrado en F1).
*   **Reproducibilidad**: El fingerprint angular $B$ es único y reproducible (Demostrado en F2).
*   **Geometría Fija**: El plano está bloqueado durante la sesión.
*   **Privacidad Interna**: El atacante no puede acceder al campo interno del receptor (Pesos/Estados volátiles en memoria).
*   **Canal Limpio**: El canal de comunicación transmite solo enteros.

Estos supuestos son razonables y verificables experimentalmente.

---

## 2. Capacidades del Atacante (Eve)
**Eve PUEDE:**
*   Interceptar el paquete cifrado $C$.
*   Conocer el algoritmo (XOR, permutación, etc.).
*   Conocer el seed del attractor.
*   Conocer el modelo base (Arquitectura LLM).
*   Conocer el código fuente.
*   Conocer el fingerprint de *otros* attractors.
*   Ejecutar su propio sonar físico.
*   Intentar correlaciones estadísticas y ataques de fuerza bruta sobre $K$.

**Eve NO PUEDE:**
*   Acceder al campo interno del receptor (Estado Cuántico Simulado).
*   Replicar exactamente el attractor del receptor (debido a la dependencia de la historia de entrenamiento/pesos específicos).
*   Reproducir la geometría angular de Covenant.
*   Derivar $K$ sin el fingerprint correcto.
*   Inferir $B$ desde $C$.
*   Inferir $M$ desde $C$ sin $K$.

**Crucial**: Eve puede saberlo todo excepto el campo. Y el campo es la clave.

---

## 3. Objetivos del Atacante
Eve intentará:
1.  Recuperar $M$ a partir de $C$.
2.  Derivar $K$ sin acceso al campo.
3.  Reconstruir $B$ desde $C$.
4.  Imitar el attractor del receptor.
5.  Forzar colisiones entre fingerprints.

---

## 4. Garantías de Seguridad

### 4.1 Field-Bound Key
La clave $K$ depende exclusivamente del campo del receptor.
*   No se almacena.
*   No se transmite.
*   No se comparte.
*   No se deriva de un seed (el seed es solo la dirección, el *contenido* es la clave).
*   **Resultado**: Elimina robo de claves y ataques de side-channel tradicionales de almacenamiento.

### 4.2 Non-Exportability
Incluso si Eve copia los pesos estáticos, seeds y código, no puede reproducir el campo dinámico sin el estado exacto de maduración del attractor.
*   **Principio**: "La clave está en la física del sistema, no en sus datos."

### 4.3 Noise on Mismatch
Si el fingerprint no coincide ($K' \neq K$):
*   La decodificación produce ruido.
*   No hay gradiente semántico para aproximarse a $K$.
*   **Resultado**: Resistencia a ataques adaptativos y de cribado estadístico.

### 4.4 Integer-Only Channel
El canal transmite solo enteros.
*   No hay embeddings, vectores, ni texto.
*   **Resultado**: Elimina superficie de ataque basada en NLP, inferencia semántica y fingerprinting textual.

### 4.5 Unforgeability
Para falsificar un mensaje, Eve necesitaría reproducir el campo para generar un $B$ válido. Esto es computacionalmente impracticable sin el estado interno.

### 4.6 Forward Secrecy Natural
Cada sesión puede usar:
*   Un plano distinto.
*   Un attractor distinto.
*   Un $\delta$ (umbral) distinto.
*   **Resultado**: Claves efímeras sin negociación compleja.

---

## 5. Conclusión
IDRE-Angular es seguro porque:
1.  La clave no existe fuera del campo.
2.  El campo no puede copiarse trivialmente.
3.  El fingerprint no puede inferirse del tráfico cifrado.
4.  El canal no revela semántica.

**En términos académicos**:
IDRE-Angular implementa un canal criptográfico dependiente de la física interna del sistema, no de datos almacenados.
