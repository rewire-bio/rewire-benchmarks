nextflow.enable.dsl=2
import groovy.json.JsonSlurper
import groovy.json.JsonOutput

process PREPARE {
    tag "${job.id}"
    cache 'deep'
    input:
    tuple val(job), path(source, stageAs: 'input/*'), path(assets, stageAs: 'assets')
    path stage_script
    output:
    tuple val(job), path('prepared'), path('assets')
    script:
    def encoded = JsonOutput.toJson(job).bytes.encodeBase64().toString()
    """
    python '${stage_script}' prepare --job-base64 '${encoded}' --assets assets
    """
}

process RUN {
    tag "${job.id}"
    cache 'deep'
    cpus { job.resources?.cpus ?: 1 }
    memory { job.resources?.memory ?: '4 GB' }
    time { job.resources?.time ?: '1h' }
    clusterOptions { params.execution_profile == 'slurm' ?
        "--account=${params.slurm_account}" + (job.resources?.gpu_count ?
        " --gres=gpu:${job.resources.gpu_type}:${job.resources.gpu_count}" : '') : null }
    containerOptions { params.execution_profile == 'slurm' && job.resources?.gpu_count ? '--nv' : '' }
    input:
    tuple val(job), path(prepared), path(assets)
    path stage_script
    output:
    tuple val(job), path(prepared), path('runs')
    script:
    def encoded = JsonOutput.toJson(job).bytes.encodeBase64().toString()
    """
    python '${stage_script}' run --job-base64 '${encoded}' --assets assets
    """
}

process EVALUATE {
    tag "${job.id}"
    cache 'deep'
    input:
    tuple val(job), path(prepared), path(runs)
    path stage_script
    output:
    tuple val(job), path('evaluated')
    script:
    def encoded = JsonOutput.toJson(job).bytes.encodeBase64().toString()
    """
    python '${stage_script}' evaluate --job-base64 '${encoded}'
    """
}

process EXPORT {
    tag "${job.id}"
    cache 'deep'
    publishDir { "${params.outdir}/${job.id}" }, mode: 'copy', overwrite: false
    input:
    tuple val(job), path(evaluated)
    path stage_script
    output:
    tuple val(job.id), path('delivery/*')
    script:
    def encoded = JsonOutput.toJson(job).bytes.encodeBase64().toString()
    """
    python '${stage_script}' export --job-base64 '${encoded}'
    """
}

workflow {
    if (!params.jobs) error 'Supply --jobs /absolute/path/jobs.json'
    if (!params.outdir) error 'Supply a new --outdir (private evaluation artifacts)'
    if (!params.environment_id) error 'Supply --environment_id with the exact native lock or image identity'
    if (params.execution_profile != 'native' && !(params.container_image ==~ /.+@sha256:[a-f0-9]{64}/))
        error 'Container profiles require --container_image registry/image@sha256:<64 hex digest>'
    if (params.execution_profile == 'google_batch') {
        if (!params.google_project || !params.google_region || !params.google_service_account || !params.google_work_dir)
            error 'Google Batch requires explicit project, region, service account and gs:// work directory'
        if (!(params.google_work_dir ==~ /gs:\/\/[a-z0-9][a-z0-9._-]+\/.+/) ||
            !(params.outdir ==~ /gs:\/\/[a-z0-9][a-z0-9._-]+\/.+/))
            error 'Google Batch work and output directories require gs://bucket/nonempty-prefix'
    }
    if (params.execution_profile == 'slurm' && (!params.slurm_partition || !params.slurm_account))
        error 'Slurm requires --slurm_partition and --slurm_account'
    def manifest = file(params.jobs, checkIfExists:true)
    def jobs = new JsonSlurper().parseText(manifest.text)
    if (!(jobs instanceof List) || jobs.isEmpty()) error 'Jobs must be a nonempty JSON array'
    if (jobs.collect{it.id}.unique().size() != jobs.size()) error 'Job IDs must be unique'
    def rejectSecrets
    rejectSecrets = { value ->
        if (value instanceof Map) value.each { key, item ->
            if (key.toString() =~ /(?i)(^|_)(token|password|secret|credential|api_key)(_|$)/)
                error 'Credentials are forbidden in job manifests'
            rejectSecrets(item)
        }
        else if (value instanceof List) value.each { rejectSecrets(it) }
    }
    rejectSecrets(jobs)
    if (params.execution_profile == 'dgx' && !(params.gpu_devices ==~ /[0-9]+(,[0-9]+)*/))
        error 'DGX requires --gpu_devices with explicitly allocated device indices'
    // Snapshot local scientific code/resources as part of every cache key.
    def scientificFiles = []
    ["${projectDir}/../../packages/rewirebench/src", "${projectDir}/../../benchmarks/mfass/src"].each { root ->
        new File(root).eachFileRecurse { entry ->
            if (entry.isFile() && !entry.path.contains('__pycache__') && !entry.name.endsWith('.pyc'))
                scientificFiles << entry
        }
    }
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    scientificFiles.sort { a, b -> a.path <=> b.path }.each { entry ->
        codeDigest.update(entry.path.substring(projectDir.toString().length()).getBytes('UTF-8'))
        codeDigest.update(entry.bytes)
    }
    def codeSnapshot = codeDigest.digest().encodeHex().toString()
    jobs.each { job ->
        if (!(job.id ==~ /[A-Za-z0-9][A-Za-z0-9._-]{0,79}/)) error 'Unsafe job ID'
        if (!job.source) error "Missing source for ${job.id}"
        def gpuCount = job.resources?.gpu_count
        if (gpuCount != null) {
            if (!(gpuCount instanceof Integer) || gpuCount < 1 || !(job.resources.gpu_type ==~ /[A-Za-z0-9][A-Za-z0-9_-]*/))
                error 'GPU requests need a positive integer gpu_count and explicit safe gpu_type'
            if (!(params.execution_profile in ['google_batch', 'slurm', 'dgx']))
                error 'GPU requests require google_batch, slurm or dgx'
            if (params.execution_profile == 'google_batch' && !params.google_install_gpu_drivers)
                error 'Google GPU jobs require explicit --google_install_gpu_drivers true'
            if (params.execution_profile == 'dgx' && gpuCount > params.gpu_devices.split(',').size())
                error 'GPU request exceeds explicitly allocated DGX devices'
        }
        job.scientific_code_sha256 = codeSnapshot
        job.environment_id = params.environment_id
        job.container_image = params.container_image ?: 'native'
        job.execution_profile = params.execution_profile
    }
    inputs = Channel.fromList(jobs).map { job ->
        def source = file(job.source, checkIfExists:true)
        def assets = file(job.assets ?: "${projectDir}/examples/assets", checkIfExists:true)
        tuple(job, source, assets)
    }
    stage_script = Channel.value(file("${projectDir}/bin/workflow_stage.py"))
    PREPARE(inputs, stage_script)
    RUN(PREPARE.out, stage_script)
    EVALUATE(RUN.out, stage_script)
    EXPORT(EVALUATE.out, stage_script)
}
