<template>
  <div class="page">
    <el-row :gutter="16">
      <!-- 账号信息 -->
      <el-col :xs="24" :md="10">
        <el-card header="账号信息">
          <div class="profile">
            <el-avatar :size="54" class="avatar">{{ avatarText }}</el-avatar>
            <div class="profile-main">
              <div class="name">{{ user?.display_name || user?.username }}</div>
              <div class="muted">{{ user?.username }}</div>
            </div>
            <el-tag size="small" :type="user?.role === 'admin' ? 'danger' : 'info'">
              {{ user?.role === 'admin' ? '管理员' : '普通用户' }}
            </el-tag>
          </div>

          <el-descriptions :column="1" border size="small" class="mt16">
            <el-descriptions-item label="用户 ID">{{ user?.id }}</el-descriptions-item>
            <el-descriptions-item label="上次登录">
              {{ fmtTime(user?.last_login_at) }}
            </el-descriptions-item>
            <el-descriptions-item label="创建时间">
              {{ fmtTime(user?.created_at) }}
            </el-descriptions-item>
            <el-descriptions-item label="密码存储">
              <el-tag size="small" type="success">bcrypt 加盐哈希</el-tag>
            </el-descriptions-item>
          </el-descriptions>

          <el-alert type="info" :closable="false" class="mt16" show-icon>
            <template #title>关于密码安全</template>
            <div class="hint">
              平台<b>不保存也无法还原</b>你的明文密码；数据库里只有 bcrypt 哈希。
              会话使用 HttpOnly Cookie，JavaScript 无法读取，可防 XSS 窃取。
            </div>
          </el-alert>
        </el-card>
      </el-col>

      <!-- 修改密码 -->
      <el-col :xs="24" :md="14">
        <el-card header="修改密码">
          <el-form ref="formRef" :model="form" :rules="rules" label-width="96px">
            <el-form-item label="原密码" prop="oldPassword">
              <el-input v-model="form.oldPassword" type="password" show-password
                        placeholder="请输入当前密码" autocomplete="current-password" />
            </el-form-item>
            <el-form-item label="新密码" prop="newPassword">
              <el-input v-model="form.newPassword" type="password" show-password
                        placeholder="至少 8 位，含两类以上字符" autocomplete="new-password" />
              <div class="pw-meter">
                <div class="pw-bars">
                  <span v-for="i in 4" :key="i" class="pw-bar"
                        :class="i <= strength.score ? strength.cls : ''" />
                </div>
                <span class="hint">{{ strength.text }}</span>
              </div>
            </el-form-item>
            <el-form-item label="确认新密码" prop="confirm">
              <el-input v-model="form.confirm" type="password" show-password
                        placeholder="再次输入新密码" autocomplete="new-password" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="saving" @click="onSubmit">
                保存并重新登录
              </el-button>
              <el-button @click="reset">重置</el-button>
            </el-form-item>
          </el-form>
        </el-card>

        <!-- 用户管理（仅管理员） -->
        <el-card v-if="isAdmin()" header="用户管理" class="mt16">
          <div class="card-head mb12">
            <span class="hint">仅管理员可见。新用户密码同样以 bcrypt 哈希存储。</span>
            <el-button size="small" type="primary" @click="dialogVisible = true">
              <el-icon><Plus /></el-icon>新建用户
            </el-button>
          </div>
          <el-table :data="users" size="small" v-loading="usersLoading">
            <el-table-column prop="id" label="ID" width="60" />
            <el-table-column prop="username" label="用户名" min-width="120" />
            <el-table-column prop="display_name" label="显示名" min-width="100" />
            <el-table-column label="角色" width="90">
              <template #default="{ row }">
                <el-tag size="small" :type="row.role === 'admin' ? 'danger' : 'info'">
                  {{ row.role === 'admin' ? '管理员' : '用户' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="上次登录" min-width="150">
              <template #default="{ row }">{{ fmtTime(row.last_login_at) }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>

    <!-- 新建用户对话框 -->
    <el-dialog v-model="dialogVisible" title="新建用户" width="420px">
      <el-form ref="newFormRef" :model="newUser" :rules="newRules" label-width="80px">
        <el-form-item label="用户名" prop="username">
          <el-input v-model="newUser.username" placeholder="3-32 位字母/数字/下划线" />
        </el-form-item>
        <el-form-item label="显示名">
          <el-input v-model="newUser.display_name" placeholder="可选" />
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input v-model="newUser.password" type="password" show-password
                    placeholder="至少 8 位，含两类以上字符" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="creating" @click="onCreateUser">创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { auth } from '../api'
import { authState, isAdmin, logout } from '../stores/auth'

const router = useRouter()
const user = computed(() => authState.user)
const avatarText = computed(() =>
  (user.value?.display_name || user.value?.username || '?').slice(0, 1).toUpperCase())

const formRef = ref()
const saving = ref(false)
const form = reactive({ oldPassword: '', newPassword: '', confirm: '' })

// 前端做一次强度提示（后端仍会强制校验，前端只是即时反馈）
const strength = computed(() => {
  const p = form.newPassword || ''
  let score = 0
  if (p.length >= 8) score++
  if (/[a-z]/.test(p) && /[A-Z]/.test(p)) score++
  if (/\d/.test(p)) score++
  if (/[^A-Za-z0-9]/.test(p)) score++
  if (p.length >= 12) score = Math.min(4, score + 1)
  const map = [
    { text: '太短', cls: 'weak' },
    { text: '较弱', cls: 'weak' },
    { text: '一般', cls: 'mid' },
    { text: '较强', cls: 'good' },
    { text: '很强', cls: 'good' },
  ]
  return { score: p ? score : 0, ...map[Math.min(score, 4)] }
})

const rules = {
  oldPassword: [{ required: true, message: '请输入原密码', trigger: 'blur' }],
  newPassword: [
    { required: true, message: '请输入新密码', trigger: 'blur' },
    { min: 8, message: '至少 8 位', trigger: 'blur' },
  ],
  confirm: [
    { required: true, message: '请再次输入新密码', trigger: 'blur' },
    {
      validator: (_r, v, cb) =>
        v === form.newPassword ? cb() : cb(new Error('两次输入的密码不一致')),
      trigger: 'blur',
    },
  ],
}

async function onSubmit() {
  try {
    await formRef.value.validate()
  } catch {
    return
  }
  saving.value = true
  try {
    await auth.changePassword(form.oldPassword, form.newPassword)
    ElMessage.success('密码已更新，请用新密码重新登录')
    await logout()
    router.replace({ name: 'login' })
  } catch (e) {
    ElMessage.error(e.message || '修改失败')
  } finally {
    saving.value = false
  }
}

function reset() {
  form.oldPassword = ''
  form.newPassword = ''
  form.confirm = ''
  formRef.value?.clearValidate()
}

// ---------------- 用户管理 ----------------
const users = ref([])
const usersLoading = ref(false)
const dialogVisible = ref(false)
const creating = ref(false)
const newFormRef = ref()
const newUser = reactive({ username: '', password: '', display_name: '' })
const newRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 8, message: '至少 8 位', trigger: 'blur' },
  ],
}

async function loadUsers() {
  if (!isAdmin()) return
  usersLoading.value = true
  try {
    users.value = await auth.users()
  } catch (e) {
    /* 静默 */
  } finally {
    usersLoading.value = false
  }
}

async function onCreateUser() {
  try {
    await newFormRef.value.validate()
  } catch {
    return
  }
  creating.value = true
  try {
    await auth.createUser({ ...newUser })
    ElMessage.success('用户已创建')
    dialogVisible.value = false
    Object.assign(newUser, { username: '', password: '', display_name: '' })
    await loadUsers()
  } catch (e) {
    ElMessage.error(e.message || '创建失败')
  } finally {
    creating.value = false
  }
}

function fmtTime(t) {
  if (!t) return '-'
  const d = new Date(t)
  if (Number.isNaN(d.getTime())) return t
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}`
}

onMounted(loadUsers)
</script>

<style scoped>
.profile { display: flex; align-items: center; gap: 12px; }
.avatar {
  background: linear-gradient(135deg, var(--brand), var(--brand-light));
  color: #fff;
  font-size: 20px;
  font-weight: 600;
}
.profile-main { flex: 1; min-width: 0; }
.profile-main .name { font-size: 15px; font-weight: 600; }

.pw-meter { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
.pw-bars { display: flex; gap: 4px; }
.pw-bar {
  width: 34px; height: 5px; border-radius: 3px;
  background: #e2e8f0; transition: background .2s;
}
.pw-bar.weak { background: #ef4444; }
.pw-bar.mid { background: #f59e0b; }
.pw-bar.good { background: #16a34a; }
</style>
