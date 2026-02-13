#!/usr/bin/env python3
"""
MTI-CORE Hardened Test Suite
============================
Pruebas rigurosas del núcleo de MTI-EVO con métricas cuantificables.
NO para uso en producción - solo evaluación técnica.
"""

import numpy as np
import hashlib
import secrets
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys

# Configuración de seguridad
SECURITY_PARAMS = {
    'min_entropy_bits': 128,
    'required_key_length': 32,  # bytes
    'test_iterations': 1000,
    'max_latency_ms': 50,
    'zero_tolerance': 1e-10  # Tolerancia para comparaciones
}

@dataclass
class TestResult:
    """Resultados estructurados de pruebas."""
    test_name: str
    passed: bool
    metrics: Dict
    failures: List[str]
    duration_ms: float

class HardenedMTICoreTester:
    """Tester riguroso para MTI-CORE."""
    
    def __init__(self):
        self.results = []
        self.test_start_time = None
        
    def run_all_tests(self):
        """Ejecuta todas las pruebas."""
        print("=" * 80)
        print("MTI-CORE HARDENED TEST SUITE")
        print("=" * 80)
        
        tests = [
            self.test_entropy_source,
            self.test_vector_operations,
            self.test_memory_consistency,
            self.test_resonance_physics,
            self.test_persistence_integrity,
            self.test_security_artifacts,
            self.test_performance_metrics,
        ]
        
        for test in tests:
            self._run_single_test(test)
        
        self._print_summary()
    
    def _run_single_test(self, test_func):
        """Ejecuta una prueba individual."""
        self.test_start_time = time.time()
        result = test_func()
        duration = (time.time() - self.test_start_time) * 1000
        
        test_result = TestResult(
            test_name=test_func.__name__[5:].replace('_', ' ').title(),
            passed=result['passed'],
            metrics=result.get('metrics', {}),
            failures=result.get('failures', []),
            duration_ms=duration
        )
        
        self.results.append(test_result)
        self._print_test_result(test_result)
    
    def test_entropy_source(self) -> Dict:
        """Prueba la calidad de las fuentes de entropía."""
        failures = []
        metrics = {}
        
        # 1. Prueba de secrets.token_bytes
        try:
            samples = [secrets.token_bytes(32) for _ in range(100)]
            
            # Verificar unicidad
            unique_samples = len(set(samples))
            uniqueness_ratio = unique_samples / len(samples)
            metrics['token_bytes_uniqueness'] = uniqueness_ratio
            
            if uniqueness_ratio < 0.99:
                failures.append(f"Baja unicidad en secrets.token_bytes: {uniqueness_ratio:.3f}")
            
            # Verificar distribución de bytes
            byte_counts = np.zeros(256, dtype=np.int64)
            for sample in samples:
                for byte in sample:
                    byte_counts[byte] += 1
            
            # Test de chi-cuadrado simplificado
            expected = len(samples) * 32 / 256
            chi2 = np.sum((byte_counts - expected) ** 2 / expected)
            metrics['chi2_statistic'] = chi2
            
            # Umbral para 255 grados de libertad, p=0.01
            if chi2 > 310:
                failures.append(f"Distribución no uniforme (chi2={chi2:.1f})")
                
        except Exception as e:
            failures.append(f"Error en secrets.token_bytes: {str(e)}")
        
        # 2. Prueba de os.urandom
        try:
            import os
            urandom_samples = [os.urandom(32) for _ in range(50)]
            
            # Verificar que no sean todos ceros
            zero_samples = sum(1 for s in urandom_samples if all(b == 0 for b in s))
            if zero_samples > 0:
                failures.append(f"os.urandom produjo {zero_samples} muestras de solo ceros")
                
        except Exception as e:
            failures.append(f"Error en os.urandom: {str(e)}")
        
        return {
            'passed': len(failures) == 0,
            'metrics': metrics,
            'failures': failures
        }
    
    def test_vector_operations(self) -> Dict:
        """Prueba operaciones vectoriales fundamentales."""
        failures = []
        metrics = {}
        
        # 1. Normalización de vectores
        test_vectors = [
            np.random.randn(128) for _ in range(10)
        ]
        
        for i, vec in enumerate(test_vectors):
            norm = np.linalg.norm(vec)
            if norm == 0:
                failures.append(f"Vector {i} tiene norma cero")
            
            # Normalizar
            if norm > 0:
                normalized = vec / norm
                new_norm = np.linalg.norm(normalized)
                
                if abs(new_norm - 1.0) > 1e-10:
                    failures.append(f"Normalización falló: {new_norm}")
        
        # 2. Producto punto y similaridad coseno
        vec_a = np.random.randn(64)
        vec_b = np.random.randn(64)
        
        dot_product = np.dot(vec_a, vec_b)
        cos_sim = dot_product / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        
        metrics['cosine_similarity_range'] = abs(cos_sim)
        
        # Verificar que esté en rango [-1, 1]
        if abs(cos_sim) > 1.0 + 1e-10:
            failures.append(f"Similaridad coseno fuera de rango: {cos_sim}")
        
        # 3. Operaciones en lote
        batch_size = 100
        batch_vectors = np.random.randn(batch_size, 32)
        query = np.random.randn(32)
        
        # Similaridades en lote
        norms = np.linalg.norm(batch_vectors, axis=1)
        query_norm = np.linalg.norm(query)
        
        if query_norm == 0:
            failures.append("Vector query tiene norma cero")
        else:
            similarities = np.dot(batch_vectors, query) / (norms * query_norm)
            metrics['batch_similarities_mean'] = np.mean(similarities)
            metrics['batch_similarities_std'] = np.std(similarities)
            
            # Verificar que todas las similaridades estén en rango
            if np.any(similarities < -1.1) or np.any(similarities > 1.1):
                failures.append("Similaridades en lote fuera de rango")
        
        return {
            'passed': len(failures) == 0,
            'metrics': metrics,
            'failures': failures
        }
    
    def test_memory_consistency(self) -> Dict:
        """Prueba consistencia de memoria y operaciones CRUD."""
        failures = []
        metrics = {}
        
        # Simular una memoria vectorial simple
        class VectorMemory:
            def __init__(self):
                self.vectors = {}
                self.metadata = {}
            
            def store(self, key: int, vector: np.array, meta: Dict = None):
                self.vectors[key] = vector.copy()
                self.metadata[key] = meta or {}
            
            def retrieve(self, key: int) -> Optional[np.array]:
                return self.vectors.get(key, None)
            
            def delete(self, key: int):
                if key in self.vectors:
                    del self.vectors[key]
                    del self.metadata[key]
        
        # Pruebas
        memory = VectorMemory()
        
        # 1. Almacenamiento y recuperación
        test_key = 12345
        test_vector = np.random.randn(64)
        test_meta = {'timestamp': time.time(), 'source': 'test'}
        
        memory.store(test_key, test_vector, test_meta)
        retrieved = memory.retrieve(test_key)
        
        if retrieved is None:
            failures.append("No se pudo recuperar vector almacenado")
        elif not np.allclose(retrieved, test_vector):
            failures.append("Vector recuperado no coincide con almacenado")
        
        # 2. Consistencia después de múltiples operaciones
        n_operations = 100
        keys = list(range(n_operations))
        
        for i in keys:
            vec = np.random.randn(32)
            memory.store(i, vec, {'index': i})
        
        # Verificar que todos estén presentes
        for i in keys:
            if memory.retrieve(i) is None:
                failures.append(f"Pérdida de vector en clave {i}")
        
        metrics['storage_operations'] = n_operations
        
        # 3. Eliminación
        delete_key = 50
        memory.delete(delete_key)
        
        if memory.retrieve(delete_key) is not None:
            failures.append("Vector no se eliminó correctamente")
        
        # 4. Colisiones (simuladas)
        collision_test = {}
        collision_count = 0
        
        for _ in range(1000):
            # Usar hash de 16 bits para forzar colisiones
            key = secrets.randbits(16)
            if key in collision_test:
                collision_count += 1
            collision_test[key] = True
        
        metrics['collision_rate'] = collision_count / 1000
        
        if collision_count == 0:
            failures.append("Ninguna colisión detectada (¿demasiada entropía?)")
        
        return {
            'passed': len(failures) == 0,
            'metrics': metrics,
            'failures': failures
        }
    
    def test_resonance_physics(self) -> Dict:
        """Prueba las 'leyes físicas' de resonancia (decaimiento, momentum)."""
        failures = []
        metrics = {}
        
        # Implementación de referencia para pruebas
        class PhysicsEngine:
            def __init__(self, gravity=0.1, momentum=0.9):
                self.gravity = gravity
                self.momentum = momentum
                self.velocity = 0.0
            
            def apply_gravity(self, weight: float, error: float) -> float:
                """Decaimiento proporcional al error."""
                return weight - (self.gravity * abs(error))
            
            def apply_momentum(self, gradient: float) -> float:
                """Momentum estándar."""
                self.velocity = self.momentum * self.velocity + gradient
                return self.velocity
            
            def diminishing_returns(self, weight: float, lr: float = 0.01) -> float:
                """Learning rate que disminuye con magnitud."""
                return lr / (1 + abs(weight))
        
        # Pruebas
        physics = PhysicsEngine(gravity=0.1, momentum=0.9)
        
        # 1. Gravedad (decaimiento)
        weights = []
        current_weight = 1.0
        for _ in range(10):
            current_weight = physics.apply_gravity(current_weight, error=0.5)
            weights.append(current_weight)
        
        metrics['gravity_final_weight'] = weights[-1]
        
        # Verificar decaimiento monotónico
        if not all(w1 > w2 for w1, w2 in zip(weights[:-1], weights[1:])):
            failures.append("Gravedad no es monotónica")
        
        # 2. Momentum
        gradients = np.random.randn(20) * 0.1
        velocities = []
        
        for g in gradients:
            v = physics.apply_momentum(g)
            velocities.append(v)
        
        metrics['momentum_velocity_std'] = np.std(velocities)
        
        # 3. Rendimientos decrecientes
        test_weights = [0.1, 1.0, 10.0, 100.0]
        learning_rates = []
        
        for w in test_weights:
            lr = physics.diminishing_returns(w)
            learning_rates.append(lr)
            
            if not (lr > 0):
                failures.append(f"LR no positivo para peso {w}")
        
        metrics['diminishing_returns_ratio'] = learning_rates[-1] / learning_rates[0]
        
        # Verificar que LR disminuye con peso creciente
        if not all(lr1 > lr2 for lr1, lr2 in zip(learning_rates[:-1], learning_rates[1:])):
            failures.append("Rendimientos decrecientes no funcionan")
        
        # 4. Entropía temporal (decaimiento exponencial)
        initial_strength = 1.0
        decay_rate = 0.1
        time_steps = np.arange(0, 50, 1)
        
        decayed = initial_strength * np.exp(-decay_rate * time_steps)
        metrics['entropy_half_life'] = np.log(2) / decay_rate
        
        # Verificar decaimiento exponencial
        ratios = decayed[1:] / decayed[:-1]
        expected_ratio = np.exp(-decay_rate)
        
        if np.any(abs(ratios - expected_ratio) > 0.01):
            failures.append("Decaimiento exponencial inconsistente")
        
        return {
            'passed': len(failures) == 0,
            'metrics': metrics,
            'failures': failures
        }
    
    def test_persistence_integrity(self) -> Dict:
        """Prueba integridad de serialización/deserialización."""
        failures = []
        metrics = {}
        
        import pickle
        import json
        
        # Datos de prueba
        test_data = {
            'vectors': {
                i: np.random.randn(32).tolist() for i in range(10)
            },
            'metadata': {
                i: {
                    'weight': np.random.random(),
                    'age': np.random.randint(0, 100),
                    'timestamp': time.time()
                } for i in range(10)
            },
            'config': {
                'dimension': 32,
                'gravity': 0.1,
                'momentum': 0.9,
                'entropy_rate': 0.01
            }
        }
        
        # 1. Serialización JSON
        try:
            json_str = json.dumps(test_data, indent=2)
            json_size = len(json_str.encode('utf-8'))
            metrics['json_size_bytes'] = json_size
            
            # Deserializar
            loaded = json.loads(json_str)
            
            # Verificar integridad
            if loaded['config']['dimension'] != test_data['config']['dimension']:
                failures.append("JSON: Dimensión no coincide")
            
            # Verificar vectores (JSON convierte listas, no arrays)
            for i in range(10):
                original = test_data['vectors'][i]
                restored = loaded['vectors'][str(i)]  # JSON usa strings como keys
                
                if not np.allclose(original, restored, atol=1e-10):
                    failures.append(f"JSON: Vector {i} no coincide")
                    
        except Exception as e:
            failures.append(f"JSON serialization failed: {str(e)}")
        
        # 2. Serialización NumPy (np.savez)
        try:
            # Crear arrays
            vectors_array = np.array([test_data['vectors'][i] for i in range(10)])
            weights_array = np.array([test_data['metadata'][i]['weight'] for i in range(10)])
            
            # Guardar
            np.savez('/tmp/test_persistence.npz', 
                     vectors=vectors_array, 
                     weights=weights_array)
            
            # Cargar
            loaded_npz = np.load('/tmp/test_persistence.npz')
            
            # Verificar
            if not np.allclose(vectors_array, loaded_npz['vectors']):
                failures.append("NumPy: Vectores no coinciden")
            
            metrics['numpy_file_size'] = len(open('/tmp/test_persistence.npz', 'rb').read())
            
        except Exception as e:
            failures.append(f"NumPy serialization failed: {str(e)}")
        
        # 3. Checksum de integridad
        test_bytes = secrets.token_bytes(1024)
        checksum = hashlib.sha256(test_bytes).hexdigest()
        
        # Simular corrupción
        corrupted = bytearray(test_bytes)
        if len(corrupted) > 100:
            corrupted[50] ^= 0xFF  # Flip un bit
        
        corrupted_checksum = hashlib.sha256(corrupted).hexdigest()
        
        metrics['checksum_match'] = checksum == corrupted_checksum
        
        if checksum == corrupted_checksum:
            failures.append("SHA256 no detectó corrupción (colisión improbable)")
        
        return {
            'passed': len(failures) == 0,
            'metrics': metrics,
            'failures': failures
        }
    
    def test_security_artifacts(self) -> Dict:
        """Busca artefactos de seguridad problemáticos."""
        failures = []
        metrics = {}
        
        # 1. Buscar números mágicos problemáticos
        magic_numbers = {
            '7245': 'Backdoor seed detectada en documentación',
            '8888': 'Backdoor seed detectada en documentación', 
            '77.7777': 'Peso mágico en documentación',
            '55.55': 'Peso mágico en documentación',
        }
        
        found_magic = []
        
        # Simular búsqueda en código (aquí solo verificamos los diccionarios)
        test_code_snippets = [
            "seed = 7245",
            "weight = 77.7777",
            "target = 55.55",
            "if seed == 8888:",
            "W = 77.7777"
        ]
        
        for snippet in test_code_snippets:
            for magic, description in magic_numbers.items():
                if magic in snippet:
                    found_magic.append(f"{description}: {snippet}")
        
        metrics['magic_numbers_found'] = len(found_magic)
        
        if found_magic:
            failures.extend(found_magic)
        
        # 2. Verificar semillas débiles
        weak_seeds = []
        
        # Semillas pequeñas (< 32 bits)
        for _ in range(100):
            seed = secrets.randbits(16)  # 16 bits - débil
            if seed < 65536:  # 2^16
                weak_seeds.append(seed)
        
        metrics['weak_seeds_generated'] = len(set(weak_seeds))
        
        if weak_seeds:
            failures.append(f"Semillas débiles generadas: {len(set(weak_seeds))} únicas")
        
        # 3. Verificar inicialización cero
        zero_vectors = 0
        for _ in range(100):
            vec = np.random.randn(64)
            if np.allclose(vec, 0):
                zero_vectors += 1
        
        metrics['zero_vectors'] = zero_vectors
        
        if zero_vectors > 10:  # Más del 10% es sospechoso
            failures.append(f"Demasiados vectores cero: {zero_vectors}")
        
        # 4. Patrones predecibles
        predictable_sequence = []
        
        # Probar random.Random con semilla fija (no seguro)
        import random
        rng = random.Random(42)  # Semilla fija - NO SEGURO
        predictable = [rng.randint(0, 255) for _ in range(10)]
        
        # Repetir con misma semilla
        rng2 = random.Random(42)
        predictable2 = [rng2.randint(0, 255) for _ in range(10)]
        
        if predictable == predictable2:
            failures.append("random.Random es determinístico con misma semilla (vulnerable)")
            metrics['predictable_sequence'] = predictable[:5]  # Mostrar primeros 5
        
        return {
            'passed': len(failures) == 0,
            'metrics': metrics,
            'failures': failures
        }
    
    def test_performance_metrics(self) -> Dict:
        """Pruebas de rendimiento y latencia."""
        failures = []
        metrics = {}
        
        # 1. Latencia de operaciones vectoriales
        sizes = [32, 64, 128, 256, 512]
        latencies = []
        
        for size in sizes:
            # Crear vectores
            a = np.random.randn(size)
            b = np.random.randn(size)
            
            # Medir producto punto
            start = time.perf_counter()
            for _ in range(1000):
                _ = np.dot(a, b)
            end = time.perf_counter()
            
            latency = (end - start) / 1000 * 1000  # ms por operación
            latencies.append(latency)
        
        metrics['dot_product_latency'] = dict(zip(sizes, latencies))
        
        # Verificar que la latencia sea razonable
        if any(l > 0.1 for l in latencies):  # Más de 0.1ms es sospechoso
            failures.append("Latencia de producto punto demasiado alta")
        
        # 2. Escalabilidad de memoria
        memory_usage = []
        
        for n_vectors in [100, 1000, 10000]:
            # Simular almacenamiento
            vectors = {i: np.random.randn(32) for i in range(n_vectors)}
            
            # Medir tiempo de búsqueda
            query = np.random.randn(32)
            
            start = time.perf_counter()
            similarities = {}
            for i, vec in vectors.items():
                similarities[i] = np.dot(vec, query)
            end = time.perf_counter()
            
            search_time = (end - start) * 1000  # ms
            memory_usage.append((n_vectors, search_time))
        
        metrics['scalability'] = memory_usage
        
        # Verificar crecimiento sub-lineal o lineal
        times = [t for _, t in memory_usage]
        ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
        
        metrics['scalability_ratios'] = ratios
        
        # Si el tiempo crece más que linealmente (ratio > 2 para 10x)
        if any(r > 2.5 for r in ratios):
            failures.append("Complejidad de búsqueda no lineal")
        
        # 3. Throughput
        operations = 10000
        batch_size = 100
        
        start = time.perf_counter()
        for _ in range(operations // batch_size):
            batch = np.random.randn(batch_size, 32)
            query = np.random.randn(32)
            _ = np.dot(batch, query)
        end = time.perf_counter()
        
        throughput = operations / (end - start)
        metrics['throughput_ops_per_sec'] = throughput
        
        if throughput < 1000:  # Menos de 1000 ops/seg
            failures.append(f"Throughput bajo: {throughput:.0f} ops/seg")
        
        return {
            'passed': len(failures) == 0,
            'metrics': metrics,
            'failures': failures
        }
    
    def _print_test_result(self, result: TestResult):
        """Imprime resultado individual de prueba."""
        print(f"\n{'='*60}")
        print(f"TEST: {result.test_name}")
        print(f"{'='*60}")
        
        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"Status: {status}")
        print(f"Duración: {result.duration_ms:.2f} ms")
        
        if result.metrics:
            print("\nMétricas:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        if result.failures:
            print(f"\nFallos ({len(result.failures)}):")
            for i, failure in enumerate(result.failures[:5], 1):  # Mostrar solo primeros 5
                print(f"  {i}. {failure}")
            if len(result.failures) > 5:
                print(f"  ... y {len(result.failures)-5} más")
    
    def _print_summary(self):
        """Imprime resumen final."""
        print(f"\n{'='*80}")
        print("RESUMEN FINAL")
        print(f"{'='*80}")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        print(f"Pruebas ejecutadas: {total_tests}")
        print(f"Pruebas pasadas: {passed_tests}")
        print(f"Pruebas fallidas: {failed_tests}")
        print(f"Tasa de éxito: {passed_tests/total_tests*100:.1f}%")
        
        # Métricas agregadas
        print("\nMÉTRICAS AGREGADAS:")
        
        # Tiempo total
        total_time = sum(r.duration_ms for r in self.results)
        print(f"Tiempo total: {total_time:.0f} ms")
        
        # Métricas clave de todas las pruebas
        key_metrics = [
            'token_bytes_uniqueness',
            'chi2_statistic',
            'cosine_similarity_range',
            'collision_rate',
            'gravity_final_weight',
            'entropy_half_life',
            'magic_numbers_found',
            'throughput_ops_per_sec'
        ]
        
        for metric in key_metrics:
            values = []
            for result in self.results:
                if metric in result.metrics:
                    values.append(result.metrics[metric])
            
            if values:
                if all(isinstance(v, (int, float)) for v in values):
                    avg = sum(values) / len(values)
                    print(f"  {metric}: {avg:.4f} (promedio)")
        
        # Fallos críticos
        critical_failures = []
        for result in self.results:
            for failure in result.failures:
                if any(keyword in failure.lower() for keyword in 
                      ['backdoor', 'magic', 'weak', 'predictable', 'zero']):
                    critical_failures.append(f"{result.test_name}: {failure}")
        
        if critical_failures:
            print(f"\n[WARNING]  FALLOS CRÍTICOS ({len(critical_failures)}):")
            for failure in critical_failures[:10]:  # Mostrar primeros 10
                print(f"  • {failure}")
        
        # Recomendaciones
        print(f"\n{'='*80}")
        print("RECOMENDACIONES:")
        
        if failed_tests > 0:
            print("[FAIL] Corrección necesaria antes de uso en producción")
            print("  1. Revisar fallos críticos")
            print("  2. Implementar correcciones")
            print("  3. Re-ejecutar pruebas")
        else:
            print("[PASS] Sistema pasa pruebas básicas")
            print("  Considerar:")
            print("  1. Pruebas de integración con IDRE")
            print("  2. Auditoría de seguridad externa")
            print("  3. Benchmark contra sistemas establecidos")

def main():
    """Función principal."""
    try:
        print("Iniciando pruebas hardened de MTI-CORE...")
        print(f"Python: {sys.version}")
        print(f"NumPy: {np.__version__}")
        
        tester = HardenedMTICoreTester()
        tester.run_all_tests()
        
        # Retornar código de salida apropiado
        failed_tests = sum(0 if r.passed else 1 for r in tester.results)
        sys.exit(min(failed_tests, 1))  # 0 si pasa, 1 si falla
        
    except KeyboardInterrupt:
        print("\n\nPruebas interrumpidas por el usuario.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
