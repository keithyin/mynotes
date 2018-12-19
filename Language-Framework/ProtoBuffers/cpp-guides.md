# protobuf in CPP

## 定义

```protobuf
syntax = "proto2";

package tutorial;

message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    required string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```

## 编译

`protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/addressbook.proto`



## 修饰符

* `requires` : 此字段的值必须被提供，否则的话 `message` 将会被认为没有被初始化，如果没有提供的话，序列化的时候不会报错，但是反序列化的时候会报错
* `optional` : 字段的值可以设置也可以不设置。如果不设置，将使用默认值（指定的默认值/类型的默认值）
* `repeated` : 此字段的值可以重复 0到N 次，顺序会被保留，可以看作一个动态数组



## protobuf API

* 编译的时候，编译器会生成这些 API
* `package name` 会变成 `namespace name{}`
* 对于 `required string name`
  * `has_name()` : 
  * `clear_name()`
  * `name()` : 访问字段
  * `set_name()`
  * `mutable_name()`
  * `release_name()`
* 对于 `optional string email`
  * `has_email()`
  * `clear_email()`
  * `email()` 
  * `set_email()`
  * `mutable_email()`
  * `release_email()`
* 对于 `repeated PhoneNumber phones`
  * `phones_size()`
  * ​

```c++
// 内嵌类型会这么处理，拿出来，加个前缀
enum Person_PhoneType {
  Person_PhoneType_MOBILE = 0,
  Person_PhoneType_HOME = 1,
  Person_PhoneType_WORK = 2
};

class Person : public ::google::protobuf::Message {
 public:
  Person();
  virtual ~Person();

  Person(const Person& from);

  inline Person& operator=(const Person& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Person& default_instance();

  void Swap(Person* other);
  // nested types ----------------------------------------------------

  typedef Person_PhoneNumber PhoneNumber;

  typedef Person_PhoneType PhoneType;
  static const PhoneType MOBILE = Person_PhoneType_MOBILE;
  static const PhoneType HOME = Person_PhoneType_HOME;
  static const PhoneType WORK = Person_PhoneType_WORK;
  static inline bool PhoneType_IsValid(int value) {
    return Person_PhoneType_IsValid(value);
  }
  static const PhoneType PhoneType_MIN =
    Person_PhoneType_PhoneType_MIN;
  static const PhoneType PhoneType_MAX =
    Person_PhoneType_PhoneType_MAX;
  static const int PhoneType_ARRAYSIZE =
    Person_PhoneType_PhoneType_ARRAYSIZE;
  static inline const ::google::protobuf::EnumDescriptor*
  PhoneType_descriptor() {
    return Person_PhoneType_descriptor();
  }
  static inline const ::std::string& PhoneType_Name(PhoneType value) {
    return Person_PhoneType_Name(value);
  }
  static inline bool PhoneType_Parse(const ::std::string& name,
      PhoneType* value) {
    return Person_PhoneType_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  // required string name = 1;
  inline bool has_name() const;
  inline void clear_name();
  static const int kNameFieldNumber = 1;
  inline const ::std::string& name() const;
  inline void set_name(const ::std::string& value);
  inline void set_name(const char* value);
  inline void set_name(const char* value, size_t size);
  inline ::std::string* mutable_name();
  inline ::std::string* release_name();
  inline void set_allocated_name(::std::string* name);

  // required int32 id = 2;
  inline bool has_id() const;
  inline void clear_id();
  static const int kIdFieldNumber = 2;
  inline ::google::protobuf::int32 id() const;
  inline void set_id(::google::protobuf::int32 value);

  // optional string email = 3;
  inline bool has_email() const;
  inline void clear_email();
  static const int kEmailFieldNumber = 3;
  inline const ::std::string& email() const;
  inline void set_email(const ::std::string& value);
  inline void set_email(const char* value);
  inline void set_email(const char* value, size_t size);
  inline ::std::string* mutable_email();
  inline ::std::string* release_email();
  inline void set_allocated_email(::std::string* email);

  // repeated .tutorial.Person.PhoneNumber phones = 4;
  inline int phones_size() const;
  inline void clear_phones();
  static const int kPhonesFieldNumber = 4;
  inline const ::tutorial::Person_PhoneNumber& phones(int index) const;
  inline ::tutorial::Person_PhoneNumber* mutable_phones(int index);
  inline ::tutorial::Person_PhoneNumber* add_phones();
  inline const ::google::protobuf::RepeatedPtrField< ::tutorial::Person_PhoneNumber >&
      phones() const;
  inline ::google::protobuf::RepeatedPtrField< ::tutorial::Person_PhoneNumber >*
      mutable_phones();

  // @@protoc_insertion_point(class_scope:tutorial.Person)
 private:
  inline void set_has_name();
  inline void clear_has_name();
  inline void set_has_id();
  inline void clear_has_id();
  inline void set_has_email();
  inline void clear_has_email();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::std::string* name_;
  ::std::string* email_;
  ::google::protobuf::RepeatedPtrField< ::tutorial::Person_PhoneNumber > phones_;
  ::google::protobuf::int32 id_;
  friend void  protobuf_AddDesc_demo_2eproto();
  friend void protobuf_AssignDesc_demo_2eproto();
  friend void protobuf_ShutdownFile_demo_2eproto();

  void InitAsDefaultInstance();
  static Person* default_instance_;
};
```





## Reflection

* 使用 `Descriptor` 和 `Reflection` 可以在运行时选择 如何 read/write message 字段的值

[https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.message#Message.GetDescriptor.details](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.message#Message.GetDescriptor.details)



```c++

```





## 参考资料

[https://developers.google.com/protocol-buffers/docs/cpptutorial](https://developers.google.com/protocol-buffers/docs/cpptutorial)