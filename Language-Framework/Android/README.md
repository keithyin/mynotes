* `Manifest.xml` 中声明的东西是 操作系统可以感知的东西。相当于操作系统级别的全局对象



# 四大组件

* `Activity`
  * 负责渲染视图
* `Service`
  * 负责后台执行一些操作
* `Broadcast`
  * 广播消息 & 接收其他应用广播的消息
* `ContentProvider`
  * 跨程序共享数据。







# 用户权限申请

对于危险权限来说：

1. 首先在 Manifest.xml 中声明一下。**如果不在 `Manifest.xml` 中声明一下的话，权限申请会失败**！！！！
2. 然后需要在代码中写权限申请逻辑

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
          package="com.keithyin.homemonitor">

    <uses-permission android:name="android.permission.CALL_PHONE" />

    <application
            ...
    </application>

</manifest>
```



```kotlin
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val callPhoneBtn = findViewById<Button>(R.id.callPhone)
        callPhoneBtn.setOnClickListener{
            if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CALL_PHONE)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CALL_PHONE), 1)
            }
            if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CALL_PHONE)
                != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "CALL_PHONE permission denied", Toast.LENGTH_SHORT).show()
            } else {
                call()
            }

        }

    }

    fun call() {
        val intent = Intent(Intent.ACTION_CALL);
        intent.data = Uri.parse("tel:10086")
        startActivity(intent)
    }

    override fun onPause() {
        super.onPause()

    }
```



# ContentProvider

> 应用之间共享数据。
>
> android4.2 以上就不允许**其他应用**  **直接读取当前应用的文件** 了。ContentProvider 相当于一个代理，用来代理当前应用程序的数据，向外提供接口。
>
> 当前应用程序的数据用 文件 或者 数据库 存储都是可以的！

1. 查询其他应用提供的数据

```kotlin
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btn = findViewById<Button>(R.id.startActivityBtn)
        btn.setOnClickListener{
            Toast.makeText(this, "hello", Toast.LENGTH_SHORT).show()

            val intent = Intent(this, CounterActivity::class.java)
            startActivity(intent)
        }

        readContacts.setOnClickListener{
            // 这里申请通讯录的读取权限
            if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_CONTACTS)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.READ_CONTACTS), 2)
            }

            if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_CONTACTS)
                != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "CALL_PHONE permission denied", Toast.LENGTH_SHORT).show()
            } else {
                doReadContacts()
            }
        }

    }


    fun doReadContacts() {
        val context = this
        // 这里读取 通讯录
        contentResolver.query(ContactsContract.CommonDataKinds.Phone.CONTENT_URI, null, null, null, null)?.apply {
            moveToNext()
            val name = getString(getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
            Toast.makeText(context, name, Toast.LENGTH_SHORT).show()
        }
    }


    override fun onPause() {
        super.onPause()

    }
}
```



2. 如何自定义 `ContentProvider`
   1. 继承 `ContentProvider` 并重写重要接口
   2. 在 `Manifest.xml` 中注册

```kotlin
class MyContentProvider : ContentProvider() {

    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<String>?): Int {
        TODO("Implement this to handle requests to delete one or more rows")
    }

    override fun getType(uri: Uri): String? {
        TODO(
            "Implement this to handle requests for the MIME type of the data" +
                    "at the given URI"
        )
    }

    override fun insert(uri: Uri, values: ContentValues?): Uri? {
        TODO("Implement this to handle requests to insert a new row.")
    }

    override fun onCreate(): Boolean {
        TODO("Implement this to initialize your content provider on startup.")
    }

    override fun query(
        uri: Uri, projection: Array<String>?, selection: String?,
        selectionArgs: Array<String>?, sortOrder: String?
    ): Cursor? {
        TODO("Implement this to handle query requests from clients.")
    }

    override fun update(
        uri: Uri, values: ContentValues?, selection: String?,
        selectionArgs: Array<String>?
    ): Int {
        TODO("Implement this to handle requests to update one or more rows.")
    }
}
```



> 这里面多了个 `provider`

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
          package="com.keithyin.homemonitor">

    <uses-permission android:name="android.permission.CALL_PHONE"/>
    <uses-permission android:name="android.permission.READ_CONTACTS"/>

    <application
            android:allowBackup="true"
            android:icon="@mipmap/ic_launcher"
            android:label="@string/app_name"
            android:roundIcon="@mipmap/ic_launcher_round"
            android:supportsRtl="true"
            android:theme="@style/Theme.HomeMonitor">
        <provider
                android:name=".MyContentProvider"
                android:authorities="com.keithyin.homemonitor.provider"
                android:enabled="true"
                android:exported="true">
        </provider>

        <activity
                android:name=".CounterActivity"
                android:label="@string/title_activity_counter"
                android:theme="@style/Theme.HomeMonitor.NoActionBar">
        </activity>
        <activity android:name=".MainActivity">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>

                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
    </application>

</manifest>
```



```kotlin
// 在其它应用里写如下代码。实际操作系统就可以通过 Manifest.xml 定位到底是访问的是哪个应用的哪个 ContentProvider 类，然后调用对应的 query 方法！！
// 这里为啥没有权限相关的问题呢？？？？
contentResolver.query(Uri.parse("content://com.keithyin.homemonitor.provider", null, null, null, null))?.apply{
	
}
```



# Broadcast （广播）

> 接收广播： `BroadcastReceiver`
>
> 发送广播：
>
> 隐式广播：在发送广播时没有指明接收应用是哪个
>
> Android8.0 后 大多数隐式广播都不允许使用 静态方式进行注册了（Manifest.xml方式）

1. 接收广播:  存在两种方式可以使一个应用可以接收到广播
   1. 在 Manifest.xml 文件中注册
   2. 在代码中动态注册：（感觉也是在动态的修改 Manifest.xml 文件，所以当应用关闭时，记得取消注册？）



无论哪种方式注册，都需要一个广播的处理类

```xml
class MyReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        // This method is called when the BroadcastReceiver is receiving an Intent broadcast.
        TODO("MyReceiver.onReceive() is not implemented")
    }
}
```



`Manifest.xml` 注册

> 首先是权限申请。
>
> intent-filter 就是：当该 intent 到来的时候，调用 `MyReceiver` 作为其处理类

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
          package="com.keithyin.homemonitor">
    <uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED"/>

    <application
            android:allowBackup="true"
            android:icon="@mipmap/ic_launcher"
            android:label="@string/app_name"
            android:roundIcon="@mipmap/ic_launcher_round"
            android:supportsRtl="true"
            android:theme="@style/Theme.HomeMonitor">
        <receiver
                android:name=".MyReceiver"
                android:enabled="true"
                android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.BOOT_COMPLETED"/>
            </intent-filter>
        </receiver>
    </application>

</manifest>
```



代码中注册（实际就是动态修改 `Manifest.xml` ?）

```kotlin
val intentFilter = IntentFilter()
intentFilter.addAction("android.intent.action.BOOT_COMPLETED")
receiver = MyReceiver()
registerReceiver(receiver, intentFilter)

// 这里注意应用关闭时一定要执行, 应该是为了防止修改 Manifest.xml 了吧
unregisterReceiver(receiver)
```





2. 发送广播

```kotlin
// 指明发送广播的 action name
// 接收方 intent-filter 中的 action 字段的 android:name 属性传入这个值就可以接收了
val intent = Intent("com.keithyin.homemonitor.MyBrodcast") 
intent.setPackage(packageName) // 这个包名到底是啥
sendBroadcast(intent) // 发送
```





# Service

> 程序后台运行的解决方案
>
> 
>
> * service 是运行在主线程中，和渲染线程一样。所以 service 不适合 **直接** 处理长时间任务
> * service 依赖创建 service 的应用程序，如果该程序被销毁，那么 service也会被销毁
> * 每个service只有一个实例！







# 碰到的错误及其解决方案

* https://stackoverflow.com/questions/55909804/duplicate-class-android-support-v4-app-inotificationsidechannel-found-in-modules/56815162





# Kotlin

## 判空辅助

```kotlin
a?.do() // .?操作符，当 a 为 null时，什么都不做，当 a不为 null时，执行 do()

val c = d ?: e // ?: 操作符，就像是 sql 中的 coalesce. 如果 d!=null, c = d, 如果 d==null, c = e

a!!.do() // 程序员保证不为 null，编译器不会做 null 指针检查。可能会引起运行时错误


a?.let{
    it.do1（）
    it.do2()
    ...
} // 当a为空时，啥都不做，a不为空时，会执行 let 里面的代码块
```

