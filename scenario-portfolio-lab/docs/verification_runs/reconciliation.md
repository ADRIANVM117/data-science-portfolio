# Reconciliación de la verificación de proveedor

## Incidente de solapamiento

En la primera ejecución se observaron dos pares de procesos Python asociados con `scenario-portfolio-lab`: procesos del entorno virtual iniciados a las 2026-10-10 21:39:07 y 21:40:40 (hora local), junto con intérpretes Python hijos iniciados en los mismos instantes. Se identificaron por PID, hora de inicio y ruta ejecutable dentro del proyecto. Se detuvieron antes de reanudar.

La evidencia demuestra solapamiento porque el archivo incremental tuvo 212 eventos para 109 `instrument_id` distintos, con 103 ocurrencias adicionales. No se pudo atribuir cada evento a un PID porque los eventos originales no registraban PID; por ello la causa exacta queda como `PENDING_PROCESS_OWNERSHIP_ATTRIBUTION`. La hipótesis de sesiones interactivas que sobrevivieron a su control es consistente con las horas y los procesos observados, pero no se trata como hecho demostrado.

## Regla de reconciliación

Sólo se reconcilian eventos cuyo manifiesto tenga el mismo `catalog_sha256` y la misma configuración de transporte/cadencia. Para cada `instrument_id`:

1. Agrupar los eventos por identificador.
2. Si todas las respuestas exitosas tienen el mismo candidato, estado y contenido sanitizado de `match`, conservar una sola conclusión (la última por `finished_at_utc`) y contarla una vez.
3. Si cambian candidato, estado, HTTP, clasificación o contenido de `match`, marcar `DUPLICATE_CONFLICT_REVIEW`; no promover ni recapturar automáticamente.
4. Un evento `ERROR` no sustituye un éxito compatible; queda como incidencia auditada. Sin éxito compatible, permanece reintentable.

La reconciliación no borra líneas JSONL; la vista consolidada se deriva de la regla anterior.

## Conteos del corte de reconciliación

El corte posterior a detener los procesos duplicados tuvo 212 líneas: 109 instrumentos únicos y 103 eventos duplicados (`212 - 109`). Todos los 109 últimos eventos únicos fueron `MATCHED`. Una ejecución limitada posterior agregó un `MATCHED` único, dejando 110 coincidencias únicas y `156 - 110 = 46` candidatos pendientes. Los conteos posteriores del manifiesto pertenecen al lote protegido que continúa sobre esos pendientes y no reescriben el corte histórico.

## Exclusión mutua comprobada

Con `.verifier.lock` presente y el proceso propietario activo, se intentó un segundo arranque sobre el mismo destino. Terminó con código 1 y el mensaje sanitizado `verification output is locked by another verifier`; el número de líneas permaneció 236 antes y después. Por tanto no envió solicitudes ni escribió eventos. El verificador no elimina un lock existente: sólo el propietario lo retira en su `finally`, de modo que un lock obsoleto no se elimina mientras su proceso siga activo.
