<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  /** 是否安全；未提供时只显示风险等级文案。 */
  safe?: boolean
  /** 风险等级或自定义文案。 */
  label?: string
}>()

const tagType = computed<'success' | 'danger' | 'info'>(() => {
  if (props.safe === true) return 'success'
  if (props.safe === false) return 'danger'
  return 'info'
})

const text = computed(() => props.label ?? (props.safe ? '安全' : '不安全'))
</script>

<template>
  <el-tag :type="tagType" size="small" effect="light">{{ text }}</el-tag>
</template>
